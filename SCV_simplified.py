import numpy as np

def create_reused_dict(M, g):
  reused = {}

  #x kernel fourier representation:
  kernel_x = np.zeros(M.shape)
  kernel_x[0, 0:2] = [1, -1]
  kernel_x = np.roll(kernel_x, -1, 1)
  dx_kernel_f = np.fft.fft2(kernel_x)

  # y kernel representation
  kernel_y = np.zeros(M.shape)
  kernel_y[0:2, 0] = [1, -1]
  kernel_y = np.roll(kernel_y, -1, 0)
  dy_kernel_f = np.fft.fft2(kernel_y)

  reused['x_kern_fourier'] = dx_kernel_f
  reused['y_kern_fourier'] = dy_kernel_f

  #laplacian 
  lap_kernel_f = dx_kernel_f.conj() * dx_kernel_f + dy_kernel_f.conj() * dy_kernel_f
  reused['lap_kernel_f'] = lap_kernel_f

  reused['g'] = g

  return reused

def bregman_update(u, w_x, w_y, b2x, b2y):
  #you should probably precompute this each outer iteration
  u_x = -u + np.roll(u, -1, axis=1)
  u_y = -u + np.roll(u, -1, axis=0)

  b2x_new = b2x + w_x - u_x
  b2y_new = b2y + w_y - u_y

  return [b2x_new, b2y_new]

def w_subproblem(params_dict, reused, u, w_x, w_y, b2x, b2y, c1, c2, M):
  mu = params_dict['mu']
  rho2 = params_dict['rho2']
  g = reused['g']

  def shrink(x, t):
    return np.sign(x)*np.maximum(np.abs(x) - t, 0.0)
  
  u_x = -u + np.roll(u, -1, axis=1)
  u_y = -u + np.roll(u, -1, axis=0)

  thresholds = mu/rho2 * g

  lhsx = u_x - b2x
  lhsy = u_y - b2y

  w_x_new = shrink(lhsx, thresholds) 
  w_y_new = shrink(lhsy, thresholds) 

  return [w_x_new, w_y_new]

def u_subproblem(params_dict, reused, u, w_x, w_y, b2x, b2y, c1, c2, M):
  r = (M - c1)**2  - (M - c2)**2
  lam = params_dict['lam']
  rho2 = params_dict['rho2']
  zeta5 = params_dict['zeta5']

  #numerator terms
  term1_x = w_x + b2x
  term1_y = w_y + b2y
  term1 = np.conj(reused['x_kern_fourier']) * np.fft.fft2(term1_x) + np.conj(reused['y_kern_fourier']) * np.fft.fft2(term1_y)
  term2 = np.fft.fft2(r)
  term3 = np.fft.fft2(u)

  #numerator
  num = rho2*term1 - lam*term2 + zeta5*term3

  #denominator
  denom = zeta5 + rho2*reused['lap_kernel_f']

  u_new = np.real(np.fft.ifft2(num / denom))

  u_new = np.clip(u_new, 0, 1)
  return u_new

def region_subproblem(params_dict, M, u, c1, c2):
  #solution of the proximal problems (40), (41) in the RKHS paper
  zeta3, zeta4, lam = params_dict["zeta3"], params_dict["zeta4"], params_dict["lam"]

  c1_new = (zeta3*c1 + 2*lam*(u * M).sum()) / (zeta3 + 2*lam*(u.sum()))
  out_u = 1 - u #region outside foreground
  c2_new = (zeta4*c2 + 2*lam*(out_u * M).sum()) / (zeta4 + 2*lam*(out_u.sum()))
  #placeholder
  return c1_new, c2_new

def segment(params_dict, M, g, init_mask=None, verbose=False):
  iter_diagnostics = {}
  H, W = M.shape

  
  m_ep = float(np.finfo(float).eps)


  #reused variables, like the fourier convolution kernel
  reused = create_reused_dict(M, g)


  #initialize
  u = init_mask.astype(float)
  c1 = (M * u).sum() / (u.sum() + m_ep)
  c2 = (M * (1 - u)).sum() / ((1 - u).sum() + m_ep)

#note - w_x does not mean the deriv wrt x of w, it means the x component of w
  w_x = -u + np.roll(u, -1, axis=1) #forward single step finite diff 
  w_y = -u + np.roll(u, -1, axis=0)

  b2x = np.zeros(w_x.shape) #zero initialized
  b2y = np.zeros(w_y.shape)

  for i in range(params_dict["iterations"]):
    #c1, c2 subproblems:
    c1, c2 = region_subproblem(params_dict, M, u, c1, c2)
    #u subproblem
    u = u_subproblem(params_dict, reused, u, w_x, w_y, b2x, b2y, c1, c2, M)
    #w subproblem
    [w_x, w_y] = w_subproblem(params_dict, reused, u, w_x, w_y, b2x, b2y, c1, c2, M)
    #b2 (split variable) update
    [b2x, b2y] = bregman_update(u, w_x, w_y, b2x, b2y)

    if(verbose):
      print(f'iter: {i},  c1: {c1}  c2: {c2}')
  return u
