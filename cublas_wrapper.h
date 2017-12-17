#pragma once

#include<cassert>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


//define vectors
namespace Deng
{
	namespace CUDA_Vec
	{

		cublasHandle_t cublas_handler;
		cublasStatus_t cublas_status = cublasCreate_v2(&cublas_handler);
		
		template <typename Field>
		class Col;

		//addition
		template<typename Field_l, typename Field_r>
		Col<Field_r> operator+(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec)//addition
		{
			const unsigned int dim = l_vec.dimension();
			assert((dim == r_vec.dimension()) && "Dimension mismatch in + (vector addition)!");

			Col<Field_r> a(dim);
			for (unsigned int i = 0; i < dim; ++i)
			{
				a[i] = l_vec[i] + r_vec[i];
			}
			return a;
		}
		//subtraction
		template<typename Field_l, typename Field_r>
		Col<Field_r> operator-(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec)//subtraction
		{
			const unsigned int dim = l_vec.dimension();
			assert((dim == r_vec.dimension()) && "Dimension mismatch in - (vector subtraction)!");

			Col<Field_r> a(dim);
			for (unsigned int i = 0; i < dim; ++i)
			{
				a[i] = l_vec[i] - r_vec[i];
			}
			return a;
		}
		//scalar multiplication
		template<typename Scalar, typename Field_prime>
		Col<Field_prime> operator*(const Scalar& k, const Col<Field_prime> & r_vec)//scalar multiplication
		{
			const unsigned int dim = r_vec.dimension();
			Col<Field_prime> a(dim);

			for (unsigned int i = 0; i < dim; ++i)
			{
				a[i] = k*r_vec[i];
			}
			return a;
		}
		//element-wise multiplication
		template<typename Field_l, typename Field_r>
		Col<Field_r> operator%(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec)//element-wise multiplication
		{
			const unsigned int dim = l_vec.dimension();
			assert((dim == r_vec.dimension()) && "Dimension mismatch in % (element-wise multiplication)!");

			Col<Field_r> a(dim);
			for (unsigned int i = 0; i < dim; ++i)
			{
				a[i] = l_vec[i] * r_vec[i];
			}
			return a;
		}
		//dot product. Field_l could be either Field_r or Scalar
		//for now only works for real/hermitian matrix!!!!!!!!!!!!!!!
		//choosing ^ is not quite appropriate
		template<typename Field_l, typename Field_r>
		Field_r operator^(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec)
		{
			const unsigned int dim = l_vec.dimension();
			assert((dim == r_vec.dimension()) && "Dimension mismatch in ^ (inner product)!");

			Field_r a = r_vec[0];
			//only work for scalars and matrices
			a = 0 * a;

			for (unsigned int i = 0; i < dim; ++i)
			{
				a += l_vec[i] * r_vec[i];
			}
			return a;
		}



		template <typename Field>
		class Col
		{
		protected:
			unsigned int _dim;
			Field* _vec;//vector of the number field Field
		public:
			Col();//default constructor, set _vec to nullptr
			Col(unsigned int dim);//constructor
			Col(const Col<Field>& c);//copy constructor
			Col(Col<Field>&& c);//move constructor

			void set_size(unsigned int dim);//in case constructor could not be used, say in an array
			unsigned int dimension() const//returns _dim
			{
				return _dim;
			}
			//destructor. 
			//make it virtual in case we need to inherit
			//virtual
			~Col();


			//overloading operators
			//member functions
			//copy assignment operator
			Col<Field> &operator=(const Col<Field> & rhs);
			//move constructor
			Col<Field> &operator=(Col<Field> &&rhs) noexcept;
			//other assignment operator
			//be careful with these operators!!!!!
			void operator+=(const Col<Field>& rhs);
			void operator-=(const Col<Field>& rhs);
			void operator*=(const Field k);//scalar multiplication
			void operator*=(const Col<Field>& b);//element-wise multiplication
														//element access
			Field& operator[](unsigned int idx)
			{
				assert(idx < _dim && "error in []!");
				return _vec[idx];
			}
			const Field& operator[](unsigned int idx) const
			{
				assert(idx < _dim && "error in []!");
				return _vec[idx];
			}


			//non-member operator
			//addition
			template<typename Field_l, typename Field_r>
			friend Col<Field_r> operator+(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec);
			//subtraction
			template<typename Field_l, typename Field_r>
			friend Col<Field_r> operator-(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec);
			//scalar multiplication
			template<typename Scalar, typename Field_prime>
			friend Col<Field_prime> operator*(const Scalar & k, const Col<Field_prime> & r_vec);
			//scalar multiplication in another order
			//ambiguous!!!!!!
			//template<typename Scalar, typename Field_prime>
			//friend Col<Field_prime> operator*(const Col<Field_prime> & r_vec, const Scalar & k);
			//element-wise multiplication
			template<typename Field_l, typename Field_r>
			friend Col<Field_r> operator%(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec);
			//inner product. Field_l could be either Field_r or Scalar
			//for now only works for real matrix!!!!!!!!!!!!!!!
			template<typename Field_l, typename Field_r>
			friend Field_r operator^(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec);


			//Field operator%(const Col<Field>& b); //inner product



			//    template <typename Field2>
			//    friend std::ostream& operator<< (std::ostream& out, const Col<Field2>& f);

		};
	}
}
