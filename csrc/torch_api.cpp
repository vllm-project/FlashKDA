#include <torch/extension.h>

#include "flash_kda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "FlashKDA Forward (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
        py::arg("scale"), py::arg("out"), py::arg("workspace"), py::arg("A_log"),
        py::arg("dt_bias"), py::arg("lower_bound"),
        py::arg("initial_state") = py::none(),
        py::arg("final_state") = py::none(),
        py::arg("cu_seqlens") = py::none());
    m.def("get_workspace_size", &get_workspace_size,
        "Get workspace size in bytes", py::arg("T_total"), py::arg("H"),
        py::arg("N") = 1);
}
