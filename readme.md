
# Cuckoo optimization algorithm
This repository implements a powerful meta-heuristic algorithm named the cuckoo optimization algorithm in `C++` programming language. The implementation inspired by the original `Matlab`  implementation and may not work as good as original implementation.

# Features
 - Fully vectorized
 - Support **parallel running**
 - Real valued optimization, appropriate for solving many problems.

# Requirements
- `armadillo`
- `mlpack`
- `C++14`
- `openmp` (optional)
# Arguments

The `COA` class is defined as

    COA(function<double(rowvec)> f, unsigned int n_par, double var_lo,double var_hi)
    

- `f` : A function which computes the cost value.
- `n_par` : The number of cost function arguments.
- `var_lo, var_hi` : The minimum and maximum values of each dimension.

The other parameters of COA can be set after constructing the class. See usage for more information.

# Usage
 - Implement a function which inputs a `rowvec` and returns a double value.

	   double ackley2(rowvec x) {
		   double result;
		   size_t n = x.n_cols;
		   double s1=0, s2=0;
		   for (size_t j=0; j<n; j++){
		       s1+=pow(x(j),2);
		       s2+=cos(2*PI*x(j));
		   }
		   result = -20 * exp(-.2 * sqrt(1.0/n * s1)) - exp(1.0/n * s2)+ 20 + exp(1);
		   return result;
	   }

- Define necessary parameters

	  unsigned int npar = 7;
      double var_lo = 0, var_hi = 1;
- Initialize the COA class

	  COA coa(ackley2, npar, var_lo, var_hi);

- Set the optional parameters

      coa.num_of_cuckoos = 25;
      coa.max_num_of_cuckoos = 50;
      coa.motion_coeff = 20;
      coa.radius_coeff = 15;
      coa.max_iter = 30;
      coa.cuckoo_pop_variance = 0;
      coa.accuracy = 0;
      coa.verbose = false;
- Run the optimizer

	  rowvec c = coa.run();
	  cout << c;

- Compile the source code
	- with parallelization

          g++ -larmadillo -O3 -std=c++14 -fPIC -larmadillo -lmlpack -fopenmp main.cpp -o tmp/coa_parallel
	- without paralleization

          g++ -larmadillo -O3 -std=c++14 -fPIC -larmadillo -lmlpack main.cpp -o tmp/coa_serial
          
# Benchmark

On the GNU/Linux operating systems, you can easily run `build.sh` file. This script compiles the `main.cpp` with previous section commands. The result on Intel Core i5 4200U with 4 logical cores (HT technology) for solving a simple integral equation is

	start compiling...  
	All files compiled successfully!  
	serial time is 9.18 sec  
	parallel time is 3.95 sec  
	speedup = 2.32
# Todo
- [ ] python bindings
- [ ] publish on pypi
- [ ] GPU support 
