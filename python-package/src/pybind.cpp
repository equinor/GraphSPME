#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "graph_spme.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_graphspme, m)
{
    m.doc() = R"pbdoc(
        GraphSPME
        ---------

        .. currentmodule:: _graphspme

        .. autosummary::
           :toctree: _generate

           prec_sparse
    )pbdoc";

    m.def("_prec_sparse", &prec_sparse, R"pbdoc(
        prec_sparse
    )pbdoc");

    m.def("_cov_shrink_spd", &cov_shrink_spd, R"pbdoc(
        cov_shrink_spd
    )pbdoc");

    m.def("_sparse_matrix_inverse", &sparse_matrix_inverse, R"pbdoc(
        sparse_matrix_inverse
    )pbdoc");

    m.def("_cov_ml", &cov_ml, R"pbdoc(
        cov_ml
    )pbdoc");

    m.def("_prec_nll", &prec_nll, R"pbdoc(
            prec_nll
    )pbdoc");

    m.def("_prec_aic", &prec_aic, R"pbdoc(
            prec_aic
    )pbdoc");

    m.def("_compute_amd_ordering", &compute_amd_ordering, R"pbdoc(
            compute_amd_ordering
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
