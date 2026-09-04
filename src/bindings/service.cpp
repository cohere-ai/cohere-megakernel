// nanobind bindings for mk::DecodeService, the long-lived decode service
// (continuous batching) declared in src/decode/abi.h. Reached from Python as
// `mk_ext.service`, driven by src/serving/session.py.
//
// GIL POLICY IS LOAD-BEARING HERE
//
// The service owns a decode thread that holds its internal mutex `m_` across
// each step and, when a token callback is installed, calls back into Python. A
// Python thread entering `set_slot` or `get_state` while holding the GIL would
// block on `m_` while the decode thread blocked on the GIL: a two-party
// deadlock. Every method below therefore releases the GIL, including the ones
// that look instantaneous. `run` and `wait_paused` block outright and would
// wedge the interpreter without it.
//
// The descriptor types themselves live in launch.cpp, since NmcDecodeServiceDesc
// embeds NmcLaunchDesc and the two must be registered together.

#include "bindings.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "decode/abi.h"

namespace nb = nanobind;

namespace mk_bindings {

void bind_service(nb::module_& m) {
    nb::class_<mk::DecodeService>(m, "DecodeService",
        "The long-lived decode service. One thread calls run() and blocks in "
        "it; other threads drive the loop through the signalling methods.")
        .def(
            "__init__",
            [](mk::DecodeService* self, const mk::NmcDecodeServiceDesc& desc) {
                nb::gil_scoped_release release;
                new (self) mk::DecodeService(desc);
            },
            nb::arg("desc"),
            "Create the decode service. It borrows every buffer the descriptor "
            "points at -- those must outlive it -- and shares ownership of its "
            "kv_handle, which therefore cannot be freed under a running decode.")

        .def(
            "run",
            [](mk::DecodeService& self) {
                nb::gil_scoped_release release;
                return self.run();
            },
            "Run the decode loop until stopped. Blocks for the lifetime of the "
            "service.\n\n"
            "Returns a status code so the caller can distinguish an ordinary "
            "failure from the watchdog-expired code, which triggers process "
            "termination. Use launch.last_error() for the message.")

        .def(
            "signal_pause",
            [](mk::DecodeService& self, bool paused) {
                nb::gil_scoped_release release;
                self.signal_pause(paused);
            },
            nb::arg("paused"),
            "Request pause/resume. Asynchronous; pair with wait_paused.")

        .def(
            "signal_stop",
            [](mk::DecodeService& self) {
                nb::gil_scoped_release release;
                self.signal_stop();
            })

        .def(
            "wait_paused",
            [](mk::DecodeService& self, int32_t timeout_ms) {
                nb::gil_scoped_release release;
                return self.wait_paused(timeout_ms);
            },
            nb::arg("timeout_ms"),
            "Block until the loop is parked, or the timeout elapses. False "
            "means it did not park.")

        .def(
            "notify",
            [](mk::DecodeService& self) {
                nb::gil_scoped_release release;
                self.notify();
            },
            "Wake the loop after changing slot state.")

        .def(
            "set_slot",
            [](mk::DecodeService& self, int32_t row, bool active,
               int32_t start_pos, int32_t gen_col, float temperature,
               uint64_t seed) {
                nb::gil_scoped_release release;
                self.set_slot(row, active, start_pos, gen_col, temperature,
                              seed);
            },
            nb::arg("row"), nb::arg("active"), nb::arg("start_pos"),
            nb::arg("gen_col"), nb::arg("temperature"), nb::arg("seed"),
            "Populate one batch row. The service must be paused.")

        .def(
            "get_state",
            [](mk::DecodeService& self) {
                std::vector<int32_t> active, gen_col, finish;
                {
                    nb::gil_scoped_release release;
                    // Sized from the service itself. The C ABI made the caller
                    // pass a batch and wrote the service's own batch size into
                    // the buffers regardless, so a stale caller-side batch
                    // overran them; batch_size() is why that is now impossible.
                    const auto count =
                        static_cast<size_t>(self.batch_size());
                    active.resize(count);
                    gen_col.resize(count);
                    finish.resize(count);
                    self.get_state(active.data(), gen_col.data(),
                                   finish.data());
                }
                return std::make_tuple(active, gen_col, finish);
            },
            "Snapshot (active, gen_col, finish) for every row, at the "
            "service's current batch size.")

        .def_prop_ro(
            "batch_size",
            [](mk::DecodeService& self) {
                nb::gil_scoped_release release;
                return self.batch_size();
            },
            "The service's current batch size, which set_geometry changes.")

        .def(
            "set_geometry",
            [](mk::DecodeService& self, const mk::NmcDecodeServiceDesc& desc) {
                nb::gil_scoped_release release;
                self.set_geometry(desc);
            },
            nb::arg("desc"),
            "Switch decode geometry (session batch size) mid-flight. The "
            "caller MUST hold the service paused and re-populate slot state "
            "afterwards.")

        .def_prop_ro(
            "kv_pressure",
            [](mk::DecodeService& self) {
                nb::gil_scoped_release release;
                return self.kv_pressure();
            },
            "Non-zero when the loop parked because the KV pool cannot satisfy "
            "the next step. An int, not a bool, because the scheduler reads "
            "the value.")

        .def(
            "resume_from_pressure",
            [](mk::DecodeService& self) {
                nb::gil_scoped_release release;
                self.resume_from_pressure();
            })

        .def(
            "close",
            [](mk::DecodeService& self) {
                nb::gil_scoped_release release;
                self.close();
            },
            "Stop the loop and drop this object's reference to it. In-flight "
            "calls hold their own, so this cannot free the service under them. "
            "Every other method raises after close().");
}

}  // namespace mk_bindings
