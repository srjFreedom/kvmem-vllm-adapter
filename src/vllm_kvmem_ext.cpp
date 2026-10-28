// pybind11 binding: exposes the kvmem contract machinery (VllmMemoryBackend / MemorySession)
// to Python. The VllmKvHandle is implemented in Python (torch knows vLLM's 5D paged-KV layout);
// this trampoline forwards each C++ handle call into the Python object. D2H/H2D buffer pointers
// are passed as addresses (uintptr) + byte count so the Python side wraps them with ctypes and
// fills/drains them in place (no copy).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdio>

#include "kvmem/memory_contract.hpp"
// v0.19's recall design keeps one statistics store per sequence: a RawKvStore holding only the
// mean-K sums (`n_layer = full-attention layers`, `n_embd_k = kv_heads*head_dim`,
// `block_tokens = 64`). The reference product program fills it from device-side bucket sums
// (backends/ninfer .../storage/memory_statistics.cpp) and scores it with the mean-K/page-softmax
// scorers, so the adapter binds it instead of keeping a second mean-key store in Python.
#include "kvmem/raw_kv_store.hpp"
// v0.19 recall surface: the Query frame (which pages are candidates, which are required) and the
// two scorers over the mean-K statistics. `mean_query_cosine_scores` is the P1 scorer; the page
// softmax reference is available without the CUDA object.
#include "kvmem/graph_recall.hpp"
#include "kvmem/page_softmax_cuda.hpp"
#include "kvmem/query_policy.hpp"
// nvme_kv_tier.hpp was removed in upstream v0.19 (shared SSD tier retired). The two Python
// call sites (kvmem_sparse_orchestrator / kvmem_history_cache) are guarded by try/except and a
// disk-config condition, so the RAM-only path degrades to self._disk = None without this binding.
#include "vllm_kvmem/vllm_backend.hpp"
#include "vllm_kvmem/vllm_sparse_selector.hpp"

namespace py = pybind11;
using namespace kvmem;
using namespace vllm_kvmem;
using Geometry = VllmKvHandle::Geometry; // nested in the handle interface

// ---------------------------------------------------------------------------
// Trampoline: C++ VllmKvHandle -> Python object.
// ---------------------------------------------------------------------------
class VllmKvHandlePy : public VllmKvHandle {
public:
    explicit VllmKvHandlePy(py::object obj) : py_self_(std::move(obj)) {}
    Geometry geometry() const override { return py::cast<Geometry>(py_self_.attr("geometry")()); }
    std::vector<CatalogBlock> catalog() const override {
        return py::cast<std::vector<CatalogBlock>>(py_self_.attr("catalog")());
    }
    std::vector<uint64_t> resident() const override {
        return py::cast<std::vector<uint64_t>>(py_self_.attr("resident")());
    }
    MemoryStamp stamp() const noexcept override {
        return py::cast<MemoryStamp>(py_self_.attr("stamp")());
    }
    void spill_block(uint64_t id, void * dst) const override {
        py_self_.attr("spill_block")(id, py::int_(reinterpret_cast<uintptr_t>(dst)), py::int_(record_bytes()));
    }
    void restore_block(uint64_t id, const void * src) override {
        py_self_.attr("restore_block")(id, py::int_(reinterpret_cast<uintptr_t>(src)), py::int_(record_bytes()));
    }
    void publish_view(const ExecutionView & view) override {
        py_self_.attr("publish_view")(py::cast(view));
    }
    // Optional async extension: only dispatched when the Python handle implements it, so the
    // blocking handles written before the async arm keep working unchanged.
    void * pinned_scratch(size_t bytes) override {
        if (!py::hasattr(py_self_, "pinned_scratch")) return VllmKvHandle::pinned_scratch(bytes);
        return reinterpret_cast<void *>(py::cast<uintptr_t>(py_self_.attr("pinned_scratch")(bytes)));
    }
    void * transfer_stream() override {
        if (!py::hasattr(py_self_, "transfer_stream")) return VllmKvHandle::transfer_stream();
        return reinterpret_cast<void *>(py::cast<uintptr_t>(py_self_.attr("transfer_stream")()));
    }
    void spill_block_async(uint64_t id, void * dst, void * stream) const override {
        if (!py::hasattr(py_self_, "spill_block_async")) {
            VllmKvHandle::spill_block_async(id, dst, stream);
            return;
        }
        py_self_.attr("spill_block_async")(id, py::int_(reinterpret_cast<uintptr_t>(dst)),
                                           py::int_(reinterpret_cast<uintptr_t>(stream)));
    }
    void restore_block_async(uint64_t id, const void * src, void * stream) override {
        if (!py::hasattr(py_self_, "restore_block_async")) {
            VllmKvHandle::restore_block_async(id, src, stream);
            return;
        }
        py_self_.attr("restore_block_async")(id, py::int_(reinterpret_cast<uintptr_t>(src)),
                                            py::int_(reinterpret_cast<uintptr_t>(stream)));
    }

private:
    uint64_t record_bytes() const {
        const auto g = geometry();
        return static_cast<uint64_t>(g.block_tokens) * g.num_kv_heads * g.head_dim *
               g.dtype_bytes * 2 * g.num_layers;
    }
    py::object py_self_;
};

static std::shared_ptr<VllmKvHandle> make_handle(py::object obj) {
    return std::make_shared<VllmKvHandlePy>(std::move(obj));
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(vllm_kvmem_ext, m) {
    m.doc() = "KVMem adapter for one vLLM TP rank (out-of-tree).";

    py::class_<SessionIdentity>(m, "SessionIdentity")
        .def(py::init<>())
        .def_readwrite("id", &SessionIdentity::id)
        .def_readwrite("incarnation", &SessionIdentity::incarnation);

    // The backend descriptor (identity + layout + capabilities), exposed so the engine and the
    // tests can read the asynchronous_transfer capability and the paged-KV layout.
    py::class_<BackendIdentity>(m, "BackendIdentity")
        .def(py::init<>())
        .def_readwrite("backend", &BackendIdentity::backend)
        .def_readwrite("revision", &BackendIdentity::revision)
        .def_readwrite("model", &BackendIdentity::model)
        .def_readwrite("state_schema", &BackendIdentity::state_schema);
    py::enum_<PayloadPlaneKind>(m, "PayloadPlaneKind")
        .value("Key", PayloadPlaneKind::Key)
        .value("Value", PayloadPlaneKind::Value)
        .value("KeyScale", PayloadPlaneKind::KeyScale)
        .value("ValueScale", PayloadPlaneKind::ValueScale)
        .export_values();
    py::class_<PayloadPlane>(m, "PayloadPlane")
        .def(py::init<>())
        .def_readwrite("layer", &PayloadPlane::layer)
        .def_readwrite("kind", &PayloadPlane::kind)
        .def_readwrite("offset", &PayloadPlane::offset)
        .def_readwrite("bytes", &PayloadPlane::bytes)
        .def_readwrite("alignment", &PayloadPlane::alignment)
        .def_readwrite("token_stride", &PayloadPlane::token_stride);
    py::class_<PayloadLayout>(m, "PayloadLayout")
        .def(py::init<>())
        .def_readwrite("format", &PayloadLayout::format)
        .def_readwrite("version", &PayloadLayout::version)
        .def_readwrite("block_tokens", &PayloadLayout::block_tokens)
        .def_readwrite("record_bytes", &PayloadLayout::record_bytes)
        .def_readwrite("alignment", &PayloadLayout::alignment)
        .def_readwrite("planes", &PayloadLayout::planes);
    py::class_<MemoryCapabilities>(m, "MemoryCapabilities")
        .def(py::init<>())
        .def_readwrite("page_tokens", &MemoryCapabilities::page_tokens)
        .def_readwrite("max_position", &MemoryCapabilities::max_position)
        .def_readwrite("native_payload", &MemoryCapabilities::native_payload)
        .def_readwrite("sparse_view", &MemoryCapabilities::sparse_view)
        .def_readwrite("original_rope_positions", &MemoryCapabilities::original_rope_positions)
        .def_readwrite("pre_rope_statistics", &MemoryCapabilities::pre_rope_statistics)
        .def_readwrite("state_checkpoint", &MemoryCapabilities::state_checkpoint)
        .def_readwrite("asynchronous_transfer", &MemoryCapabilities::asynchronous_transfer);
    // The sparse-selection layer (KVMem "A mode"): wraps kvmem's pick_topk_blocks and expands
    // the selected 128-token selection blocks into vLLM's 16-token pages.
    py::class_<VllmSparseSelector::Config>(m, "SparseSelectorConfig")
        .def(py::init<>())
        .def_readwrite("select_block_tokens", &VllmSparseSelector::Config::select_block_tokens)
        .def_readwrite("budget_tokens", &VllmSparseSelector::Config::budget_tokens)
        .def_readwrite("gen_budget_tokens", &VllmSparseSelector::Config::gen_budget_tokens)
        .def_readwrite("sink_tokens", &VllmSparseSelector::Config::sink_tokens)
        .def_readwrite("recent_tokens", &VllmSparseSelector::Config::recent_tokens)
        .def_readwrite("page_tokens", &VllmSparseSelector::Config::page_tokens)
        .def_readwrite("lrfu", &VllmSparseSelector::Config::lrfu)
        .def_readwrite("lrfu_lambda", &VllmSparseSelector::Config::lrfu_lambda)
        .def_readwrite("lrfu_top_fraction", &VllmSparseSelector::Config::lrfu_top_fraction);
    py::class_<VllmSparseSelector>(m, "VllmSparseSelector")
        .def(py::init<VllmSparseSelector::Config>())
        .def("set_context_tokens", &VllmSparseSelector::set_context_tokens)
        .def("set_block_scores", &VllmSparseSelector::set_block_scores)
        .def("select_pages", &VllmSparseSelector::select_pages)
        .def("select_prefill_pressure_pages",
             &VllmSparseSelector::select_prefill_pressure_pages,
             py::arg("mandatory_pages") = std::vector<uint32_t>{})
        .def("budget_pages", &VllmSparseSelector::budget_pages)
        .def("selection_blocks", &VllmSparseSelector::selection_blocks)
        .def("tokens_per_selection_block", &VllmSparseSelector::tokens_per_selection_block)
        .def("pages_per_selection_block", &VllmSparseSelector::pages_per_selection_block);
    py::class_<MemoryDescriptor>(m, "MemoryDescriptor")
        .def(py::init<>())
        .def_readwrite("identity", &MemoryDescriptor::identity)
        .def_readwrite("layout", &MemoryDescriptor::layout)
        .def_readwrite("capabilities", &MemoryDescriptor::capabilities);

    py::class_<MemoryStamp>(m, "MemoryStamp")
        .def(py::init<>())
        .def_readwrite("valid", &MemoryStamp::valid)
        .def_readwrite("session", &MemoryStamp::session)
        .def_readwrite("history_revision", &MemoryStamp::history_revision)
        .def_readwrite("view_version", &MemoryStamp::view_version)
        .def_readwrite("frontier", &MemoryStamp::frontier)
        .def_readwrite("execution_history", &MemoryStamp::execution_history);

    py::class_<BlockDescriptor>(m, "BlockDescriptor")
        .def(py::init<>())
        .def_readwrite("session", &BlockDescriptor::session)
        .def_readwrite("id", &BlockDescriptor::id)
        .def_readwrite("original_start", &BlockDescriptor::original_start)
        .def_readwrite("valid_tokens", &BlockDescriptor::valid_tokens)
        .def_readwrite("content_version", &BlockDescriptor::content_version)
        .def_readwrite("statistics_version", &BlockDescriptor::statistics_version);

    py::class_<ViewBlock>(m, "ViewBlock")
        .def(py::init<>())
        .def_readwrite("block", &ViewBlock::block)
        .def_readwrite("compact_start", &ViewBlock::compact_start);

    py::class_<ExecutionView>(m, "ExecutionView")
        .def(py::init<>())
        .def_readwrite("stamp", &ExecutionView::stamp)
        .def_readwrite("blocks", &ExecutionView::blocks)
        .def_readwrite("visible_tokens", &ExecutionView::visible_tokens);

    py::class_<Geometry>(m, "Geometry")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
            py::arg("num_layers"), py::arg("num_kv_heads"), py::arg("head_dim"),
            py::arg("block_tokens"), py::arg("dtype_bytes") = 2)
        .def_readwrite("num_layers", &Geometry::num_layers)
        .def_readwrite("num_kv_heads", &Geometry::num_kv_heads)
        .def_readwrite("head_dim", &Geometry::head_dim)
        .def_readwrite("block_tokens", &Geometry::block_tokens)
        .def_readwrite("dtype_bytes", &Geometry::dtype_bytes);

    py::class_<CatalogBlock>(m, "CatalogBlock")
        .def(py::init<>())
        .def_readwrite("session", &CatalogBlock::session)
        .def_readwrite("id", &CatalogBlock::id)
        .def_readwrite("original_start", &CatalogBlock::original_start)
        .def_readwrite("valid_tokens", &CatalogBlock::valid_tokens)
        .def_readwrite("content_version", &CatalogBlock::content_version)
        .def_readwrite("statistics_version", &CatalogBlock::statistics_version);

    py::class_<BlockCopies>(m, "BlockCopies")
        .def(py::init<>())
        .def_readwrite("block", &BlockCopies::block)
        .def_readwrite("host_version", &BlockCopies::host_version);

    py::class_<MemorySnapshot>(m, "MemorySnapshot")
        .def(py::init<>())
        .def_readwrite("stamp", &MemorySnapshot::stamp)
        .def_readwrite("blocks", &MemorySnapshot::blocks)
        .def_readwrite("resident", &MemorySnapshot::resident);

    py::class_<MemoryBudget>(m, "MemoryBudget")
        .def(py::init<uint64_t, uint64_t, uint64_t, uint64_t>(),
            py::arg("selected_tokens"), py::arg("reserve_tokens"),
            py::arg("device_tokens"), py::arg("host_payload_bytes"))
        .def_readwrite("selected_tokens", &MemoryBudget::selected_tokens)
        .def_readwrite("reserve_tokens", &MemoryBudget::reserve_tokens)
        .def_readwrite("device_tokens", &MemoryBudget::device_tokens)
        .def_readwrite("host_payload_bytes", &MemoryBudget::host_payload_bytes);

    py::class_<WorkingSetPlan>(m, "WorkingSetPlan")
        .def_readonly("next", &WorkingSetPlan::next)
        .def_readonly("mandatory", &WorkingSetPlan::mandatory)
        .def_readonly("evict", &WorkingSetPlan::evict)
        .def_readonly("spill", &WorkingSetPlan::spill)
        .def_readonly("restore", &WorkingSetPlan::restore)
        .def_readonly("device_tokens_required", &WorkingSetPlan::device_tokens_required)
        .def_readonly("host_payload_bytes_required", &WorkingSetPlan::host_payload_bytes_required);

    py::enum_<MemoryPhase>(m, "MemoryPhase")
        .value("Idle", MemoryPhase::Idle)
        .value("Prepared", MemoryPhase::Prepared)
        .value("Spilling", MemoryPhase::Spilling)
        .value("Restoring", MemoryPhase::Restoring)
        .value("Ready", MemoryPhase::Ready)
        .value("Published", MemoryPhase::Published)
        .value("Invalid", MemoryPhase::Invalid)
        .value("Detached", MemoryPhase::Detached)
        .export_values();

    py::enum_<TransferStatus>(m, "TransferStatus")
        .value("Pending", TransferStatus::Pending)
        .value("Complete", TransferStatus::Complete)
        .value("Failed", TransferStatus::Failed)
        .export_values();

    // The backend is Python-owned (shared_ptr holder) so it outlives the session.
    py::class_<VllmMemoryBackend, std::shared_ptr<VllmMemoryBackend>>(m, "VllmMemoryBackend")
        .def(py::init([](py::object handle, uint64_t pool_capacity, std::string name,
                         bool async_transfer) {
            return new VllmMemoryBackend(make_handle(std::move(handle)), pool_capacity,
                std::move(name), async_transfer);
        }), py::arg("handle"), py::arg("pool_capacity_bytes"), py::arg("backend_name"),
           py::arg("async_transfer") = false)
        .def("descriptor", &VllmMemoryBackend::descriptor)
        .def("snapshot", &VllmMemoryBackend::snapshot)
        .def("stamp", &VllmMemoryBackend::stamp)
        .def("invalidate", &VllmMemoryBackend::invalidate);

    // KVMem's host tier: capacity-enforced RAM for spilled KV payloads (the H budget).
    py::class_<kvmem::HostKvStorage>(m, "HostKvStorage")
        .def(py::init<uint64_t>(), py::arg("capacity_bytes"))
        .def("allocate", [](kvmem::HostKvStorage & self, size_t bytes) {
            return self.allocate(bytes);
        }, py::arg("bytes"))
        .def("charge", &kvmem::HostKvStorage::charged_bytes)
        .def("capacity", &kvmem::HostKvStorage::capacity_bytes);

    py::class_<kvmem::HostKvRecord>(m, "HostKvRecord")
        .def(py::init<>())
        .def("size", &kvmem::HostKvRecord::size)
        .def("empty", &kvmem::HostKvRecord::empty)
        .def("write", [](kvmem::HostKvRecord & self, py::bytes payload) {
            std::string data = payload;
            if (data.size() != self.size())
                throw std::runtime_error("payload size does not match the record");
            self.write(0, data.data(), data.size());
        }, py::arg("payload"))
        .def("read", [](const kvmem::HostKvRecord & self) {
            std::string out(self.size(), '\0');
            if (!out.empty()) self.read(0, out.data(), out.size());
            return py::bytes(out);
        })
        .def("seal", &kvmem::HostKvRecord::seal)
        .def("resident_bytes", &kvmem::HostKvRecord::resident_bytes);

    // Pure planning: lets the sparse runtime take its attention view from kvmem itself instead of
    // building one by hand (the convergence the user asked for).
    m.def("plan_working_set", [](const MemoryDescriptor & descriptor,
                                 const MemorySnapshot & snapshot,
                                 const std::vector<uint64_t> & selected,
                                 const std::vector<uint64_t> & mandatory,
                                 const MemoryBudget & budget) {
        return kvmem::plan_working_set(descriptor, snapshot, selected, mandatory, budget);
    }, py::arg("descriptor"), py::arg("snapshot"), py::arg("selected"),
       py::arg("mandatory"), py::arg("budget"));

    py::class_<MemorySession, std::unique_ptr<MemorySession>>(m, "MemorySession")
        .def(py::init([](std::shared_ptr<VllmMemoryBackend> backend) {
            return std::make_unique<MemorySession>(std::static_pointer_cast<MemoryBackend>(backend));
        }), py::arg("backend"))
        .def("prepare",
            [](MemorySession & self, py::object expected, py::object selected, py::object mandatory, py::object budget) {
                MemoryStamp stamp = expected.cast<MemoryStamp>();
                std::vector<uint64_t> sel = selected.cast<std::vector<uint64_t>>();
                std::vector<uint64_t> man = mandatory.cast<std::vector<uint64_t>>();
                MemoryBudget bud = budget.cast<MemoryBudget>();
                return self.prepare(stamp, sel, man, bud);
            })
        .def("prepare_reselect",
            [](MemorySession & self, py::object expected, VllmSparseSelector & policy,
               py::object mandatory, py::object budget) {
                auto stamp = expected.cast<MemoryStamp>();
                auto man = mandatory.cast<std::vector<uint32_t>>();
                auto bud = budget.cast<MemoryBudget>();
                return self.prepare_reselect(stamp, policy.store(), man, bud);
            },
            py::arg("expected"), py::arg("policy"), py::arg("mandatory"), py::arg("budget"))
        .def("advance", &MemorySession::advance)
        .def("publish", &MemorySession::publish)
        .def("abort", &MemorySession::abort)
        .def("phase", &MemorySession::phase)
        .def("disconnect", &MemorySession::disconnect);
    // --- mean-K statistics store (v0.19 recall design) ---------------------------------------
    py::class_<kvmem::RawKvStore, std::unique_ptr<kvmem::RawKvStore>>(m, "RawKvStore")
        .def(py::init([](uint32_t n_layer, uint32_t n_embd_k, uint32_t block_tokens) {
            kvmem::RawKvStoreConfig cfg;
            cfg.n_layer = n_layer;
            cfg.n_embd_k = n_embd_k;   // kv_heads * head_dim (the key width, not the query width)
            cfg.block_tokens = block_tokens ? block_tokens : 64;
            return std::make_unique<kvmem::RawKvStore>(cfg);
        }), py::arg("n_layer"), py::arg("n_embd_k"), py::arg("block_tokens") = 64)
        .def("write_layer_mean_sum",
            [](kvmem::RawKvStore & self, uint32_t pos0, uint32_t n, uint32_t layer,
               py::array_t<float, py::array::c_style | py::array::forcecast> sum) {
                const auto info = sum.request();
                if (info.ndim != 1 || uint64_t(info.shape[0]) != self.config().n_embd_k)
                    throw std::invalid_argument("mean-sum width must equal n_embd_k");
                self.write_layer_mean_sum(pos0, n, layer,
                                          static_cast<const float *>(info.ptr));
            }, py::arg("pos0"), py::arg("n"), py::arg("layer"), py::arg("sum"))
        .def("mean_k", [](const kvmem::RawKvStore & self, uint32_t block_id, uint32_t layer) {
            const auto width = self.config().n_embd_k;
            py::array_t<float> out(width);
            self.mean_k(block_id, layer, out.mutable_data());
            return out;
        }, py::arg("block_id"), py::arg("layer"))
        .def("n_tokens", &kvmem::RawKvStore::n_tokens, py::arg("block_id"))
        .def("has_block", &kvmem::RawKvStore::has_block, py::arg("block_id"))
        .def("truncate_to", &kvmem::RawKvStore::truncate_to, py::arg("token_pos"))
        .def("clear", &kvmem::RawKvStore::clear)
        .def("allocated_bytes", &kvmem::RawKvStore::allocated_bytes)
        .def("mean_checkpoint", &kvmem::RawKvStore::mean_checkpoint, py::arg("token_pos"))
        .def("restore_mean_checkpoint", &kvmem::RawKvStore::restore_mean_checkpoint,
             py::arg("token_pos"), py::arg("state"))
        .def_property_readonly("n_layer", [](const kvmem::RawKvStore & self) {
            return self.config().n_layer; })
        .def_property_readonly("n_embd_k", [](const kvmem::RawKvStore & self) {
            return self.config().n_embd_k; })
        .def_property_readonly("block_tokens", [](const kvmem::RawKvStore & self) {
            return self.config().block_tokens; });

    // --- Query frame and scorers (v0.19 recall) ----------------------------------------------
    py::class_<kvmem::QueryScoreFrame>(m, "QueryScoreFrame")
        .def_readonly("starts", &kvmem::QueryScoreFrame::starts)
        .def_readonly("counts", &kvmem::QueryScoreFrame::counts)
        .def_readonly("score_mask", &kvmem::QueryScoreFrame::score_mask)
        .def_readonly("transport_mask", &kvmem::QueryScoreFrame::transport_mask)
        .def_readonly("accepted_through", &kvmem::QueryScoreFrame::accepted_through);

    m.def("query_score_frame", &kvmem::query_score_frame,
          py::arg("accepted_through"), py::arg("block_tokens"), py::arg("budget_blocks"),
          py::arg("sink_blocks"), py::arg("recent_blocks"), py::arg("fixed_blocks"));

    m.def("query_prefix_required", &kvmem::query_prefix_required,
          py::arg("prefix"), py::arg("block"), py::arg("sink"), py::arg("recent"),
          py::arg("media"));

    // One scorer entry point for both aggregation modes; the per-layer arrays are borrowed for the
    // duration of the call (PageQueryView carries their pointers, it does not own them). A layer
    // passed as None is ABSENT (recurrent layers, the MTP layer), never a zero sample.
    auto score_with_view = [](const kvmem::RawKvStore & keys, py::object layers, uint32_t rows,
                              uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
                              uint32_t blocks, bool max_agg, py::object frame) {
        using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
        std::vector<Array> held;                     // owns the data for the call's duration
        std::vector<const float *> ptrs;
        std::vector<uint32_t> counts;
        std::vector<std::pair<size_t, size_t>> slots;   // (layer index, held index)
        const auto layer_count = static_cast<size_t>(py::len(layers));
        ptrs.reserve(layer_count);
        counts.reserve(layer_count);
        size_t layer = 0;
        for (py::handle item : layers) {
            if (item.is_none()) {
                // ABSENT layer (recurrent, MTP): no rows, never a zero-valued sample.
                ptrs.push_back(nullptr);
                counts.push_back(0);
            } else {
                auto array = py::cast<Array>(item);
                const auto info = array.request();
                // The core allows a layer to carry FEWER rows than `rows` (`layer_rows[l] <= rows`):
                // layers can have different capture windows, and the row count is what the scorer
                // weights by. Requiring equality here rejected such a frame outright ("query rows
                // must be [rows, query_heads, head_dim]") and dropped retrieval for the request.
                if (info.ndim != 3 || uint32_t(info.shape[1]) != query_heads ||
                    uint32_t(info.shape[2]) != head_dim || uint32_t(info.shape[0]) > rows ||
                    uint32_t(info.shape[0]) == 0)
                    throw std::invalid_argument("query rows must be [<=rows, query_heads, head_dim]");
                held.push_back(std::move(array));
                ptrs.push_back(nullptr);             // filled once `held` stops moving
                counts.push_back(uint32_t(info.shape[0]));
                slots.emplace_back(layer, held.size() - 1);
            }
            ++layer;
        }
        for (const auto & [at, index] : slots)
            ptrs[at] = static_cast<const float *>(held[index].request().ptr);
        kvmem::PageQueryView view{};
        view.data = nullptr;
        view.layer_rows = counts.data();
        view.layers = static_cast<uint32_t>(layer_count);
        view.rows = rows;
        view.query_heads = query_heads;
        view.kv_heads = kv_heads;
        view.head_dim = head_dim;
        view.layer_data = ptrs.data();
        const auto aggregation = max_agg ? kvmem::GqaScoreAggregation::Max
                                         : kvmem::GqaScoreAggregation::Mean;
        if (frame.is_none())
            return kvmem::mean_query_cosine_scores(keys, view, blocks, aggregation);
        const auto & score_frame = frame.cast<const kvmem::QueryScoreFrame &>();
        return kvmem::page_softmax_scores(keys, view, score_frame, aggregation);
    };
    // The upstream DEFAULT scorer (v0.19 resolve_query_scoring: page-softmax implies mean
    // aggregation and LRFU off). Its device storage is bounded to one layer and <=64 MiB of logits
    // instead of L*T*H*B, and score() takes a proper QueryScoreFrame (candidates, score_mask and the
    // accepted extent), not a bare block count.
    py::class_<kvmem::PageSoftmaxCuda, std::unique_ptr<kvmem::PageSoftmaxCuda>>(m, "PageSoftmaxCuda")
        .def(py::init<int>(), py::arg("device"))
        .def_static("workspace_bytes", &kvmem::PageSoftmaxCuda::workspace_bytes,
                    py::arg("max_blocks"), py::arg("query_heads"), py::arg("kv_heads"),
                    py::arg("head_dim"))
        .def("device_bytes", &kvmem::PageSoftmaxCuda::device_bytes)
        .def("score", [](kvmem::PageSoftmaxCuda & self, const kvmem::RawKvStore & keys,
                         py::object layers, uint32_t rows, uint32_t query_heads, uint32_t kv_heads,
                         uint32_t head_dim, const kvmem::QueryScoreFrame & frame, bool max_agg) {
            using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
            std::vector<Array> held;
            std::vector<const float *> ptrs;
            std::vector<uint32_t> counts;
            std::vector<std::pair<size_t, size_t>> slots;
            const auto layer_count = static_cast<size_t>(py::len(layers));
            size_t layer = 0;
            for (py::handle item : layers) {
                if (item.is_none()) {
                    ptrs.push_back(nullptr);
                    counts.push_back(0);
                } else {
                    auto array = py::cast<Array>(item);
                    const auto info = array.request();
                    if (info.ndim != 3 || uint32_t(info.shape[1]) != query_heads ||
                        uint32_t(info.shape[2]) != head_dim || uint32_t(info.shape[0]) > rows ||
                        uint32_t(info.shape[0]) == 0)
                        throw std::invalid_argument("query rows must be [<=rows, query_heads, head_dim]");
                    held.push_back(std::move(array));
                    ptrs.push_back(nullptr);
                    counts.push_back(uint32_t(info.shape[0]));
                    slots.emplace_back(layer, held.size() - 1);
                }
                ++layer;
            }
            for (const auto & [at, index] : slots)
                ptrs[at] = static_cast<const float *>(held[index].request().ptr);
            kvmem::PageQueryView view{};
            view.layer_rows = counts.data();
            view.layers = static_cast<uint32_t>(layer_count);
            view.rows = rows;
            view.query_heads = query_heads;
            view.kv_heads = kv_heads;
            view.head_dim = head_dim;
            view.layer_data = ptrs.data();
            return self.score(keys, view, frame,
                              max_agg ? kvmem::GqaScoreAggregation::Max
                                      : kvmem::GqaScoreAggregation::Mean);
        }, py::arg("keys"), py::arg("layer_rows"), py::arg("rows"), py::arg("query_heads"),
           py::arg("kv_heads"), py::arg("head_dim"), py::arg("frame"),
           py::arg("max_aggregation") = false);

    m.def("query_scores", score_with_view,
          py::arg("keys"), py::arg("layer_rows"), py::arg("rows"), py::arg("query_heads"),
          py::arg("kv_heads"), py::arg("head_dim"), py::arg("blocks"), py::arg("max_aggregation"),
          py::arg("frame") = py::none());

    // --- NVMe KV tier (disk layer, D budget) -------------------------------------------------
    // Removed with upstream v0.19: the shared SSD tier (nvme_kv_tier.hpp) was retired and the
    // RAM-only PinnedKvTier does not expose this binding. The Python call sites are guarded, so
    // a missing NvmeKvTier degrades to self._disk = None (the current RAM-only behaviour).
}
