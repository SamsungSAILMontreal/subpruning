#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "SubmodularFunction.h"
#include "functions/kernels/RBFKernel.h"
#include "functions/kernels/Kernel.h"
#include "functions/IVM.h"
#include "functions/FastIVM.h"
#include "Greedy.h"
#include "Random.h"
#include "SieveStreaming.h"
#include "SieveStreamingPP.h"
#include "ThreeSieves.h"
#include "Salsa.h"
#include "IndependentSetImprovement.h"

namespace py = pybind11;

template <class T>
class PySubmodularFunction : public SubmodularFunction<T> {
public:
    data_t operator()(std::vector<T> const &solution) const override {
        PYBIND11_OVERRIDE_PURE_NAME(
            data_t,                     /* Return type */
            SubmodularFunction<T>,         /* Parent class */
            "__call__",                 /* Name of method in Python */
            operator(),                 /* Name of function in C++ */
            solution                    /* Argument(s) */
        );
    }

    data_t peek(std::vector<T> const &cur_solution, T const &x, unsigned int pos) override {
       PYBIND11_OVERRIDE_PURE(
           data_t,
           SubmodularFunction<T>,
           peek,
           cur_solution, 
           x,
           pos
       );
    }

    void update(std::vector<T> const &cur_solution, T const &x, unsigned int pos) override {
        PYBIND11_OVERRIDE_PURE(
           void,
           SubmodularFunction<T>,
           update,
           cur_solution, 
           x,
           pos
       );
    }

    ~PySubmodularFunction() {
    }

    // See https://github.com/pybind/pybind11/issues/1049
    std::shared_ptr<SubmodularFunction<T>> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = std::make_shared<py::object>(cloned);
        auto ptr = cloned.template cast<PySubmodularFunction<T>*>();

        std::shared_ptr<SubmodularFunction<T>> newobj = std::shared_ptr<SubmodularFunction<T>>(keep_python_state_alive, ptr);

        // aliasing shared_ptr: points to `A_trampoline* ptr` but refcounts the Python object
        return newobj;
    }
};


class PyKernel : public Kernel {
public:
    data_t operator()(std::vector<data_t> const &x1, std::vector<data_t> const &x2) const override {
        PYBIND11_OVERRIDE_PURE_NAME(
            data_t,                     /* Return type */
            Kernel,                     /* Parent class */
            "__call__",                 /* Name of method in Python */
            operator(),                 /* Name of function in C++ */
            x1,                         /* Argument(s) */
            x2
        );
    }

    ~PyKernel() {}

    // See https://github.com/pybind/pybind11/issues/1049
    std::shared_ptr<Kernel> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = std::make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PyKernel*>();

        std::shared_ptr<Kernel> newobj = std::shared_ptr<Kernel>(keep_python_state_alive, ptr);

        // aliasing shared_ptr: points to `A_trampoline* ptr` but refcounts the Python object
        return newobj;
    }
};

// data_t fit_greedy_on_ivm(unsigned int K, data_t sigma, data_t scale, data_t epsilon, std::vector<std::vector<data_t>> const &X) {
//     FastIVM fastIVM(K, RBFKernel(sigma, scale), epsilon);
//     Greedy greedy(K, fastIVM);
//     greedy.fit(X);
//     return greedy.get_fval();
// }

// data_t fit_greedy_on_ivm_2(unsigned int K, data_t sigma, data_t scale, data_t epsilon) {
//     return 1.0;
//     // FastIVM fastIVM(K, RBFKernel(sigma, scale), epsilon);
//     // Greedy greedy(K, fastIVM);
//     // greedy.fit(X);
//     // return greedy.get_fval();
// }

// PYBIND11_MAKE_OPAQUE(std::vector<data_t>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<data_t>>);

template<class T>
void declare_template_classes(py::module &m, const std::string &typestr) {
    std::string pyclass_name = std::string("SubmodularFunction") + typestr;
    py::class_<SubmodularFunction<T>, PySubmodularFunction<T>, std::shared_ptr<SubmodularFunction<T>>>(m, pyclass_name.c_str())
        .def(py::init<>())
        .def("peek", &SubmodularFunction<T>::peek, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("update", &SubmodularFunction<T>::update, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("__call__", &SubmodularFunction<T>::operator())
        .def("clone", &SubmodularFunction<T>::clone, py::return_value_policy::reference);

    pyclass_name = std::string("Greedy") + typestr;
    py::class_<Greedy<T>>(m, pyclass_name.c_str())
        //.def(py::init<unsigned int, std::shared_ptr<SubmodularFunction>>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, SubmodularFunction<T>&>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<T> const &)> >(), py::arg("K"), py::arg("f"))
        .def("get_solution", &Greedy<T>::get_solution)
        .def("get_ids", &Greedy<T>::get_ids)
        .def("get_fval", &Greedy<T>::get_fval)
        .def("get_num_candidate_solutions", &Greedy<T>::get_num_candidate_solutions)
        .def("get_num_elements_stored", &Greedy<T>::get_num_elements_stored)
        .def("fit", py::overload_cast<std::vector<T> const &, unsigned int>(&Greedy<T>::fit), py::arg("X"), py::arg("iterations") = 1)
        .def("fit", py::overload_cast<std::vector<T> const &, std::vector<idx_t> const &, unsigned int>(&Greedy<T>::fit), py::arg("X"), py::arg("ids"), py::arg("iterations") = 1)
        .def("get_f", &Greedy<T>::get_f, py::return_value_policy::reference);

    pyclass_name = std::string("Random") + typestr;
    py::class_<Random<T>>(m, pyclass_name.c_str())
        .def(py::init<unsigned int, SubmodularFunction<T>&, unsigned long>(), py::arg("K"), py::arg("f"), py::arg("seed")= 0)
        .def(py::init<unsigned int, std::function<data_t (std::vector<T> const &)>, unsigned long>(), py::arg("K"), py::arg("f"), py::arg("seed") = 0)
        .def("get_solution", &Random<T>::get_solution)
        .def("get_ids", &Random<T>::get_ids)
        .def("get_fval", &Random<T>::get_fval)
        .def("get_num_candidate_solutions", &Random<T>::get_num_candidate_solutions)
        .def("get_num_elements_stored", &Random<T>::get_num_elements_stored)
        .def("fit", py::overload_cast<std::vector<T> const &, unsigned int>(&Random<T>::fit), py::arg("X"), py::arg("iterations") = 1)
        .def("fit", py::overload_cast<std::vector<T> const &, std::vector<idx_t> const &, unsigned int>(&Random<T>::fit), py::arg("X"), py::arg("ids"), py::arg("iterations") = 1)
        .def("next", &Random<T>::next, py::arg("x"), py::arg("id") = std::nullopt)
        .def("get_f", &Random<T>::get_f, py::return_value_policy::reference);

    pyclass_name = std::string("IndependentSetImprovement") + typestr;
    py::class_<IndependentSetImprovement<T>>(m, pyclass_name.c_str())
        .def(py::init<unsigned int, SubmodularFunction<T>&>(), py::arg("K"), py::arg("f"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<T> const &)>>(), py::arg("K"), py::arg("f"))
        .def("get_solution", &IndependentSetImprovement<T>::get_solution)
        .def("get_ids", &IndependentSetImprovement<T>::get_ids)
        .def("get_fval", &IndependentSetImprovement<T>::get_fval)
        .def("get_num_candidate_solutions", &IndependentSetImprovement<T>::get_num_candidate_solutions)
        .def("get_num_elements_stored", &IndependentSetImprovement<T>::get_num_elements_stored)
        .def("fit", py::overload_cast<std::vector<T> const &, unsigned int>(&IndependentSetImprovement<T>::fit), py::arg("X"), py::arg("iterations") = 1)
        .def("fit", py::overload_cast<std::vector<T> const &, std::vector<idx_t> const &, unsigned int>(&IndependentSetImprovement<T>::fit), py::arg("X"), py::arg("ids"), py::arg("iterations") = 1)
        .def("next", &IndependentSetImprovement<T>::next, py::arg("x"), py::arg("id") = std::nullopt)
        .def("get_f", &IndependentSetImprovement<T>::get_f, py::return_value_policy::reference);

    pyclass_name = std::string("SieveStreaming") + typestr;
    py::class_<SieveStreaming<T>>(m, pyclass_name.c_str())
        .def(py::init<unsigned int, SubmodularFunction<T>&, data_t, data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<T> const &)>, data_t, data_t>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"))
        .def("get_solution", &SieveStreaming<T>::get_solution)
        .def("get_ids", &SieveStreaming<T>::get_ids)
        .def("get_fval", &SieveStreaming<T>::get_fval)
        .def("get_num_candidate_solutions", &SieveStreaming<T>::get_num_candidate_solutions)
        .def("get_num_elements_stored", &SieveStreaming<T>::get_num_elements_stored)
        .def("fit", py::overload_cast<std::vector<T> const &, unsigned int>(&SieveStreaming<T>::fit), py::arg("X"), py::arg("iterations") = 1)
        .def("fit", py::overload_cast<std::vector<T> const &, std::vector<idx_t> const &, unsigned int>(&SieveStreaming<T>::fit), py::arg("X"), py::arg("ids"), py::arg("iterations") = 1)
        .def("next", &SieveStreaming<T>::next, py::arg("x"), py::arg("id") = std::nullopt)
        .def("get_f", &SieveStreaming<T>::get_f, py::return_value_policy::reference);

    pyclass_name = std::string("SieveStreamingPP") + typestr;
    py::class_<SieveStreamingPP<T>>(m, pyclass_name.c_str() )
        .def(py::init<unsigned int, SubmodularFunction<T>&, data_t, data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<T> const &)>, data_t, data_t>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"))
        .def("get_solution", &SieveStreamingPP<T>::get_solution)
        .def("get_ids", &SieveStreamingPP<T>::get_ids)
        .def("get_fval", &SieveStreamingPP<T>::get_fval)
        .def("get_num_candidate_solutions", &SieveStreamingPP<T>::get_num_candidate_solutions)
        .def("get_num_elements_stored", &SieveStreamingPP<T>::get_num_elements_stored)
        .def("fit", py::overload_cast<std::vector<T> const &, unsigned int>(&SieveStreamingPP<T>::fit), py::arg("X"), py::arg("iterations") = 1)
        .def("fit", py::overload_cast<std::vector<T> const &, std::vector<idx_t> const &, unsigned int>(&SieveStreamingPP<T>::fit), py::arg("X"), py::arg("ids"), py::arg("iterations") = 1)
        .def("next", &SieveStreamingPP<T>::next, py::arg("x"), py::arg("id") = std::nullopt)
        .def("get_f", &SieveStreamingPP<T>::get_f, py::return_value_policy::reference);

    pyclass_name = std::string("ThreeSieves") + typestr;
    py::class_<ThreeSieves<T>>(m, pyclass_name.c_str())
        .def(py::init<unsigned int, SubmodularFunction<T>&, data_t, data_t, std::string const &, unsigned int>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"), py::arg("strategy"), py::arg("T"))
        .def(py::init<unsigned int, std::function<data_t (std::vector<T> const &)>, data_t, data_t, std::string const &, unsigned int>(), py::arg("K"), py::arg("f"),  py::arg("m"), py::arg("epsilon"), py::arg("strategy"), py::arg("T"))
        .def("get_solution", &ThreeSieves<T>::get_solution)
        .def("get_ids", &ThreeSieves<T>::get_ids)
        .def("get_fval", &ThreeSieves<T>::get_fval)
        .def("get_num_candidate_solutions", &ThreeSieves<T>::get_num_candidate_solutions)
        .def("get_num_elements_stored", &ThreeSieves<T>::get_num_elements_stored)
        .def("fit", py::overload_cast<std::vector<T> const &, unsigned int>(&ThreeSieves<T>::fit), py::arg("X"), py::arg("iterations") = 1)
        .def("fit", py::overload_cast<std::vector<T> const &, std::vector<idx_t> const &, unsigned int>(&ThreeSieves<T>::fit), py::arg("X"), py::arg("ids"), py::arg("iterations") = 1)
        .def("next", &ThreeSieves<T>::next, py::arg("x"), py::arg("id") = std::nullopt)
        .def("get_f", &ThreeSieves<T>::get_f, py::return_value_policy::reference);

    pyclass_name = std::string("Salsa") + typestr;
    py::class_<Salsa<T>>(m, pyclass_name.c_str())
        .def(py::init<unsigned int, SubmodularFunction<T>&, data_t, data_t, data_t, data_t, data_t, data_t, data_t, data_t,data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"), py::arg("hilow_epsilon") = 0.05, py::arg("hilow_beta") = 0.1, py::arg("hilow_delta") = 0.025, py::arg("dense_beta") = 0.8, py::arg("dense_C1") = 10, py::arg("dense_C2") = 0.2, py::arg("fixed_epsilon") = 1.0/6.0)
        .def(py::init<unsigned int, std::function<data_t (std::vector<T> const &)>, data_t, data_t, data_t, data_t, data_t, data_t, data_t, data_t,data_t>(), py::arg("K"), py::arg("f"), py::arg("m"), py::arg("epsilon"), py::arg("hilow_epsilon") = 0.05, py::arg("hilow_beta") = 0.1, py::arg("hilow_delta") = 0.025, py::arg("dense_beta") = 0.8, py::arg("dense_C1") = 10, py::arg("dense_C2") = 0.2, py::arg("fixed_epsilon") = 1.0/6.0)
        .def("get_solution", &Salsa<T>::get_solution)
        .def("get_ids", &Salsa<T>::get_ids)
        .def("get_fval", &Salsa<T>::get_fval)
        .def("get_num_candidate_solutions", &Salsa<T>::get_num_candidate_solutions)
        .def("get_num_elements_stored", &Salsa<T>::get_num_elements_stored)
        .def("fit", py::overload_cast<std::vector<T> const &, unsigned int>(&Salsa<T>::fit), py::arg("X"), py::arg("iterations") = 1)
        .def("fit", py::overload_cast<std::vector<T> const &, std::vector<idx_t> const &, unsigned int>(&Salsa<T>::fit), py::arg("X"), py::arg("ids"), py::arg("iterations") = 1)
        .def("get_f", &Salsa<T>::get_f, py::return_value_policy::reference);

}

PYBIND11_MODULE(PySSM, m) {
    // m.def("fit_greedy_on_ivm", &fit_greedy_on_ivm, 
    //     py::arg("K"), 
    //     py::arg("sigma"),
    //     py::arg("scale"),
    //     py::arg("epsilon"),
    //     py::arg("X")
    // );

    // m.def("fit_greedy_on_ivm_2", &fit_greedy_on_ivm_2, 
    //     py::arg("K"), 
    //     py::arg("sigma"),
    //     py::arg("scale"),
    //     py::arg("epsilon")
    // );

    // py::bind_vector<std::vector<data_t>>(m, "Vector");
    // py::bind_vector<std::vector<std::vector<data_t>>>(m, "Matrix");

// Add a scoped redirect for your noisy code
    py::add_ostream_redirect(m, "ostream_redirect");

    declare_template_classes<int>(m, "int");
    declare_template_classes<std::pair<std::string, int>>(m, "pair");
    declare_template_classes<std::vector<data_t>>(m, "list");

    py::class_<Kernel, PyKernel, std::shared_ptr<Kernel>>(m, "Kernel")
        .def(py::init<>())
        .def("__call__", &Kernel::operator())
        .def("clone", &Kernel::clone, py::return_value_policy::reference);

    py::class_<RBFKernel, Kernel, std::shared_ptr<RBFKernel>>(m, "RBFKernel")
        .def(py::init<data_t, data_t>(), py::arg("sigma") = 1.0, py::arg("scale") = 1.0)
        .def(py::init<data_t>(), py::arg("sigma") = 1.0)
        .def(py::init<>())
        .def("__call__", &RBFKernel::operator())
        .def("clone", &RBFKernel::clone, py::return_value_policy::reference);

    py::class_<IVM, SubmodularFunction<std::vector<data_t>>, std::shared_ptr<IVM> >(m, "IVM")
        //.def(py::init<std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)>, data_t>(), py::arg("kernel"), py::arg("sigma"))
        .def(py::init<Kernel const &, data_t>(), py::arg("kernel"), py::arg("sigma") = 1.0)
        .def("peek", &IVM::peek, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("update", &IVM::update, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("__call__", &IVM::operator())
        .def("clone", &IVM::clone, py::return_value_policy::reference);

    py::class_<FastIVM, IVM, SubmodularFunction<std::vector<data_t>>, std::shared_ptr<FastIVM> >(m, "FastIVM")
        //.def(py::init<unsigned int, std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)>, data_t>(),  py::arg("K"),  py::arg("kernel"), py::arg("sigma"))
        .def(py::init<unsigned int, Kernel const &, data_t>(),  py::arg("K"),  py::arg("kernel"), py::arg("sigma") = 1.0)
        .def("peek", &FastIVM::peek, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("update", &FastIVM::update, py::arg("cur_solution"), py::arg("x"), py::arg("pos"))
        .def("__call__", &FastIVM::operator())
        .def("clone", &FastIVM::clone, py::return_value_policy::reference);

   }
