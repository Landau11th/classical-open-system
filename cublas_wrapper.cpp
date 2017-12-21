#include"cublas_wrapper.h"


using namespace Deng::CUDA_Vec;

//explicit instantiations

//template class Col<int>;
template class Col<float>;
//template class Col<double>;
//template class Col<std::complex<float> >;
//template class Col<std::complex<double> >;

//#define DENG_VECTOR_COMPLEX //to determine whether the field is complex (aim for inner product)
template <typename Field>
Col<Field>::Col()//default constructor, set _vec to nullptr
{
	_dim = 0;
	_vec = nullptr;
}
template <typename Field>
Col<Field>::Col(unsigned int dim)
{
	_dim = dim;
	cudaMalloc((void **)& _vec, _dim * sizeof(Field));
}
template <typename Field>
Col<Field>::Col(const Col<Field>& c)//copy constructor
{
	_dim = c.dimension();
	cudaMalloc((void **)& _vec, _dim * sizeof(Field));
	//stopped here. Cannot resolven the pointer problem
	cublasScopy(cublas_handler, _dim, c._vec, 1, _vec, 1);
}
template <typename Field>
Col<Field>::Col(Col<Field>&& c)//move constructor
{
	_dim = c.dimension();
	_vec = c._vec;

	c._vec = nullptr;
}
template <typename Field>
void Col<Field>::set_size(unsigned int dim)
{
	//if the size is already dim, there is no need to change
	if (_dim != dim)
	{
		_dim = dim;
		cudaFree(_vec);
		cudaMalloc((void **)& _vec, _dim * sizeof(Field));
	}
}
template <typename Field>
Col<Field>::~Col()
{
	//std::cout << "releasing " << _vec << std::endl;
	cudaFree(_vec);
	_vec = nullptr;
}
template <typename Field>
Col<Field>& Col<Field>::operator=(const Col<Field> & b)
{
	set_size(b.dimension());

	cublasScopy(cublas_handler, _dim, b._vec, 1, _vec, 1);

	return *this;
}
template <typename Field>
Col<Field>& Col<Field>::operator=(Col<Field> &&rhs) noexcept
{
	assert((this != &rhs) && "Memory clashes in operator '=&&'!\n");
	cudaFree(this->_vec);
	this->_vec = rhs._vec;
	rhs._vec = nullptr;

	return *this;
}
template <typename Field>
void Col<Field>::operator+=(const Col<Field>& b)
{
	/*
	unsigned int dim = this->dimension();
	Col<Field> a(dim);
	if(dim==b.dimension())
	{
	for(unsigned int i = 0; i < dim; ++i)
	{
	this->_vec[i] += b[i];
	}
	}
	else
	{
	std::cout << "Dimension mismatch in operator '+='!" << std::endl;
	}
	*/
	//here neglect all the dimension check etc
	//since +=, -=, *= are usually used for optimal performance
	for (unsigned int i = 0; i < this->_dim; ++i)
	{
		this->_vec[i] += b[i];
	}
}
template <typename Field>
void Col<Field>::operator-=(const Col<Field>& b)
{
	/*
	int dim = this->dimension();
	Col<Field> a(dim);
	if(dim==b.dimension())
	{
	int i;
	for(i=0; i<dim; i++)
	{
	this->_vec[i] -= b[i];
	}
	}
	else
	{
	std::cout << "Dimension mismatch in operator '-='!" << std::endl;
	}
	*/
	//here neglect all the dimension check etc
	//since +=, -=, *= are usually used for optimal performance
	for (unsigned int i = 0; i < this->_dim; ++i)
	{
		this->_vec[i] -= b[i];
	}
}
template <typename Field>
void Col<Field>::operator*=(const Field k)
{
	//here neglect all the dimension check etc
	//since +=, -=, *= are usually used for optimal performance
	for (unsigned int i = 0; i < this->_dim; ++i)
	{
		this->_vec[i] *= k;
	}
}
template <typename Field>
void Col<Field>::operator*=(const Col<Field>& b)
{
	//here neglect all the dimension check etc
	//since +=, -=, *= are usually used for optimal performance
	for (unsigned int i = 0; i < this->_dim; ++i)
	{
		this->_vec[i] *= b[i];
	}
}


//#ifdef DENG_VECTOR_COMPLEX
//template <typename Field>
//Field Col<Field>::operator%(const Col<Field>& b)
//{
//
//    int dddim = this->dimension();
//    Field a = 0.0;
//
//    if(dddim==b.dimension())
//    {
//        int i;
//
//        for(i=0; i<dddim; i++)
//        {
//            a += std::conj(this->vec[i] )* b.vec[i];
//        }
//    }
//    else
//    {
//        printf("Error in operator '*'!\n");
//    }
//
//    return a;
//}
//#else
///*
//template <typename Field>
//Field Col<Field>::operator%(const Col<Field>& b)
//{
//    if((typeid(Field)==typeid(arma::Mat<float>))
//    || (typeid(Field)==typeid(arma::Mat<double>))
//    || (typeid(Field)==typeid(arma::Mat<std::complex<float> >))
//    || (typeid(Field)==typeid(arma::Mat<std::complex<double> >))
//      )
//    {
//        std::cout << "Inner product is not defined for block vectors" << std::endl;
//        return 0;
//    }
//    else
//    {
//        int dddim = this->dimension();
//        Field a = 0;
//
//        if(dddim==b.dimension())
//        {
//            int i;
//
//            for(i=0; i<dddim; i++)
//            {
//                a += this->_vec[i]* b[i];
//            }
//        }
//        else
//        {
//            printf("Error in operator '*'!\n");
//        }
//
//        return a;
//    }
//}
//*/
//#endif // Col_COMPLEX
//vector operations
template <typename Field>
Field dot_product(Col<Field> a, Col<Field> b)
{
	int dim = a.dimension();
	Field dot;

	if (dim == b.dimension())
	{
		int i;

		dot = 0.0;

		for (i = 0; i<dim; i++)
		{
			dot += a[i] * b[i];
		}
	}
	else
	{
		printf("Error in dot product!\n");
	}

	return dot;
}
/*
template <typename Field>
std::ostream& operator<<(std::ostream& out, const Col<Field>& f)
{
for(int i = 0; i < f.dimension(); ++i)
{
out << f[i];
}
return out << std::endl;
}
*/