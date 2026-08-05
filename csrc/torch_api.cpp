#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "flash_kda.h"

STABLE_TORCH_LIBRARY(flash_kda, m) {
    m.def("get_workspace_size(int T_total, int H, int N=1) -> int");
    m.def(
        "fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "float scale, Tensor(a!) out, Tensor(c!) workspace, Tensor A_log, "
        "Tensor dt_bias, float lower_bound, Tensor? initial_state=None, "
        "Tensor(b!)? final_state=None, Tensor? cu_seqlens=None, "
        "bool use_vsplit=False) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flash_kda, CUDA, m) {
    m.impl("fwd", TORCH_BOX(&fwd));
}

STABLE_TORCH_LIBRARY_IMPL(flash_kda, CompositeExplicitAutograd, m) {
    m.impl("get_workspace_size", TORCH_BOX(&get_workspace_size));
}

static struct PyModuleDef flash_kda_module = {
    PyModuleDef_HEAD_INIT,
    "_C",
    nullptr,
    -1,
    nullptr,
};

PyMODINIT_FUNC PyInit__C(void) {
    return PyModule_Create(&flash_kda_module);
}
