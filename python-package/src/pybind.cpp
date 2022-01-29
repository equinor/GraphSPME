#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "graph_spme.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(graph_spme, m) {
    m.doc() = R"pbdoc(
        GraphSPME
        ---------

        .. currentmodule:: graph_spme

        .. autosummary::
           :toctree: _generate

           prec_sparse
    )pbdoc";

    m.def("prec_sparse", &prec_sparse, R"pbdoc(
        prec_sparse
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
