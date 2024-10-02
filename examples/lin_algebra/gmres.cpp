
/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <arrayfire.h>
#include <af/util.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
using namespace af;

struct info_solver {
  std::vector<float> iter;
  std::vector<float> time;
  std::vector<float> nnz;
};
template <class T>
void Update(array& x, const int k, const std::vector<T>&H,
            const int R, const std::vector<T>&s,
            const array v)
{
  T y[k+1];
  for (int i=0; i<k+1; i++)
    y[i] = s[i];
  // Back substituation, 
  // solve y, where H*y = s
  for (int i = k; i >= 0; i--)
  {
    y[i] = y[i]/ H[i + i*(R+1)];
    for (int j=i-1; j >= 0; j--)
    {
      y[j] = y[j] - H[j + i*(R+1)] * y[i];
    }
  }
  // x = v*y
  for (int j = 0; j <= k; j++){   
    x+=v.col(j)*y[j];
  }
}
template<class T>
void ApplyPlaneRotation(T &dx, T &dy, T &cs, T &sn)
{
  T temp  =  cs * dx + sn * dy;
  dy = -sn * dx + cs * dy;
  dx = temp;
}template<class T>
void GeneratePlaneRotation(T &dx, T &dy, T &cs, T &sn)
{
  if (dy == 0.0) {
    cs = 1.0;
    sn = 0.0;
  } else if (abs(dy) > abs(dx)) {
    T temp = dx / dy;
    sn = 1.0 / sqrt( 1.0 + temp*temp );
    cs = temp * sn;
  } else {
    T temp = dy / dx;
    cs = 1.0 / sqrt( 1.0 + temp*temp );
    sn = temp * cs;
  }
}
template<class T>
int GMRES(array& x, const array& A, const array& b, const int n, const int R, const T tol, const int max_iter, dtype dt) {  
  array w(n, dt);
  std::vector<T> H((R+1)* R);
  std::vector<T> s(R+1);
  std::vector<T> cs(R+1);
  std::vector<T> sn(R+1);
  array v( n, R+1, dt);
  array temp_arr(n,dt);
  T normb = norm(b);
  array r = b - matmul(A, x);
  T beta = norm(r);
  if (beta < tol) {
    return 0;
  }
  int iter = 0;
  while ( iter< max_iter )
  {
    v.col(0) = r / beta;
    std::fill(s.begin(), s.end(), 0);
    
    s[0] = beta;
    for ( int i=0; i< R; i++ )
    {
      iter++;
      // XXLiu: w = solve(A * v[i]);
      w = matmul(A, v.col(i));

      int h_idx;
      for (int k=0; k<=i; k++)
      {
        //H[i,k] =w.v[k];
        h_idx=i*(R+1)+k;
        temp_arr =  v.col(k);
        T aa = dot<T>(w, temp_arr);
        H[h_idx]=aa;      
        w-=H[h_idx]*v.col(k);
      }
      h_idx = i*(R+1)+i+1;
      T norm_h_idx=norm(w);
      H[h_idx] = norm_h_idx;
      v.col(i+1) = (1.0/H[h_idx])*w;
      for (int k = 0; k < i; k++)
        ApplyPlaneRotation<T>( H[k+i*(R+1)], H[(k+1)+i*(R+1)], cs[k], sn[k]);
      GeneratePlaneRotation<T>( H[i+i*(R+1)], H[(i+1)+i*(R+1)], cs[i], sn[i]);
      ApplyPlaneRotation<T>( H[i+i*(R+1)], H[(i+1)+i*(R+1)], cs[i], sn[i]);
      ApplyPlaneRotation<T>( s[i], s[(i+1)], cs[i], sn[i]);
      if (abs(s[i+1])/normb < tol){
        Update<T>(x, i, H, R, s, v);
        return iter;
      }    
    }
    Update<T>(x, R-1, H, R, s, v); 
    r = b -matmul(A,x);
    beta = norm(r);  
     
    if ((beta / normb) < tol) {
      return iter;
    }
  }
 return iter;
}
template <class T>
static void solve_demo(dtype dt, info_solver &solve_info) { 
  /*
  solving laplace equation
  f_xx + f_yy = 0 : x in (0,1), y in (0,1)
  with the below Dirichlet boundary condition
  f(x,0) = 4 - x^2,
  f(x,1) = 3 -x^2,
  f(0,y) = 4 - y^2,
  f(1,y) = 3 -y^2.
  The exact solution is f(x,y) = 4 - x^2 + y^2
  The domain is discretized into nxn grid points
  with a central difference scheme, the discretized equation is as folows:
  (f(x+dx,y) + f(x-dx,y) + f(x,y+dy) + f(x,y-dy) - 4f(x,y))/dx^2 = 0
  where dx=dy=1/(n-1).
  This results in a 5 diagonal matrix
  For instance, for n =3, the matrix is 9x9 will look like this:
   _                                                    _
  | 1.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00 |  
  | 0.00  1.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00 | 
  | 0.00  0.00  1.00  0.00  0.00  0.00  0.00  0.00  0.00 | 
  | 0.00  0.00  0.00  1.00  0.00  0.00  0.00  0.00  0.00 | 
  | 0.00  -.25  0.00  -.25  1.00  -.25  0.00  -.25  0.00 | 
  | 0.00  0.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00 | 
  | 0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.00  0.00 | 
  | 0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.00 | 
  |_0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00_|
  */
  printf("Solving 2D Laplace equation with GMRES method\n");
  for (int n = 20; n<400; n*=2){

    int m = n*n;
    std::vector<T> b_host(m);
    std::vector<T> x_exact(m);
    std::vector<T> A_host;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
  
    printf("matrix size %dx %d: with %d nonzero elements.\n",n,n,(n-2)*(n-2)*5+4*(n-1));
    
    T dx = 1./(n-1);
    // generate the matrix in the COO format.
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        int idx = i * n + j;
        int row_id = idx;
        T xl = i*dx;
        T yl = j*dx;
        x_exact[idx] =  ((xl-.5)*(xl-0.5) - (yl-0.5)*(yl-0.5));
        if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
          
          A_host.push_back(1);
          row_indices.push_back(row_id);
          col_indices.push_back(row_id);
        }
        else
        {   
          row_indices.push_back(row_id);
          col_indices.push_back(row_id-n);
          A_host.push_back(-1.0);
          row_indices.push_back(row_id);
          col_indices.push_back(row_id-1);
          A_host.push_back(-1.0);
          row_indices.push_back(row_id); 
          col_indices.push_back(row_id);
          A_host.push_back(4.0);
          row_indices.push_back(row_id);
          col_indices.push_back(row_id+1);
          A_host.push_back(-1.0);
          row_indices.push_back(row_id);
          col_indices.push_back(row_id+n);
          A_host.push_back(-1.0);
        }
      } 
    }
  
    
    int num_values= A_host.size();
    array val(num_values, A_host.data());
    val = val.as(dt);
    array row_idx(num_values, row_indices.data());
    array col_idx(num_values, col_indices.data());
    array sparse_mat = sparse(m, m, val, row_idx, col_idx, AF_STORAGE_COO);
    array sparse_CSR = sparseConvertTo(sparse_mat, AF_STORAGE_CSR);
    
    array xx( m, x_exact.data());
    array b = matmul(sparse_CSR, xx);
    //array x = constant(0.0, m, 1, dt);
    array x =b;
    double tol =1.e-8;
    int max_iter = 10000; 
    int R =20;
    af::sync();
    timer::start();
    int num_iter=GMRES<T>( x, sparse_CSR, b, m, R, tol, max_iter, dt); 
    af::sync();
    double solve_time = timer::stop();
    array X_exact(m, x_exact.data());
    printf("norm of error %le with  %d  iterations in %f seconds\n", norm(x-X_exact), num_iter, solve_time);
    //copy info for post processing
    solve_info.iter.push_back(static_cast<float> (num_iter));
    solve_info.time.push_back(static_cast<float> (solve_time));
    solve_info.nnz.push_back(static_cast<float> (num_values));
  }
}
int main(int argc, char **argv) {
  // usage:  iterative solve: solve_demo (device) (console on/off) (precision f32/f16)
  int device   = argc > 1 ? atoi(argv[1]) : 0;
  std::string dts = argc > 2 ? argv[2] : "f64";
  dtype dt        =f64;
  info_solver solve_info;
  if (dts == "f32")
      dt = f32;
  else if (dts != "f32" && dts !="f64") {
      std::cerr << "Unsupported datatype " << dts << ". Supported: f32 or f64"
                << std::endl;
      return EXIT_FAILURE;
  }
  try {
      af::setDevice(device);
      af::info();
      
      
      if (dt==f64){
        solve_demo<double>( dt, solve_info);
      }else{
        solve_demo<float>( dt, solve_info);
      }
      af::Window myWindow(800, 800, "2D Laplace Eqn solution: time vs nnz");
      myWindow.setAxesTitles("number of nonzeros", "Time (s)");
 
      array X( solve_info.nnz.size(), solve_info.nnz.data());
      array Y( solve_info.time.size(), solve_info.time.data());
      while (!myWindow.close()) {
        myWindow.grid(1, 2);
        myWindow.plot(X, Y);
     }
       

  } catch (af::exception &ae) { std::cerr << ae.what() << std::endl; }
  
  return 0;
}