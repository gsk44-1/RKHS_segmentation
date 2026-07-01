from scipy.stats import norm
import numpy as np
from curvature_watershed_v2 import compute_ct


def curv_map(img_input):
    C_map, S_map = compute_ct(img_input, sigma=3.0)

    S_mean, S_std = -np.pi/4, np.pi/8
    S_resp = norm.pdf(S_map, loc=S_mean, scale=S_std)
    S_resp = S_resp * np.sqrt(2*np.pi)*S_std# normalize so that the peak is at 1

    C_mean, C_std = 0, 200
    C_gaus = norm.pdf(C_map, loc=C_mean, scale=C_std)
    C_gaus = C_gaus * np.sqrt(2*np.pi)*C_std# normalize so that the peak is at 1
    C_resp = (1 - C_gaus)


    curv_boundaries = (C_resp)*S_resp

    return curv_boundaries

def create_reused_dict_(M):
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


  return reused

def bregman_update_(u, w_x, w_y, b2x, b2y):
  #you should probably precompute this each outer iteration
  u_x = -u + np.roll(u, -1, axis=1)
  u_y = -u + np.roll(u, -1, axis=0)

  b2x_new = b2x + w_x - u_x
  b2y_new = b2y + w_y - u_y

  return [b2x_new, b2y_new]

def w_subproblem_(params_dict, reused, u, w_x, w_y, b2x, b2y, c1, c2, M, g):
  mu = params_dict['mu']
  rho2 = params_dict['rho2']

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

def u_subproblem_(params_dict, reused, u, w_x, w_y, b2x, b2y, c1, c2, M, B, g):
  r = (M - c1)**2  - (M - c2)**2
  lamF = params_dict['lamF']
  lamB = params_dict['lamB']
  rho2 = params_dict['rho2']
  zeta5 = params_dict['zeta5']
  eta = params_dict['eta']
  delta = params_dict['delta']

  #numerator terms
  term1_x = w_x + b2x
  term1_y = w_y + b2y
  term1 = np.conj(reused['x_kern_fourier']) * np.fft.fft2(term1_x) + np.conj(reused['y_kern_fourier']) * np.fft.fft2(term1_y)
  term2 = np.fft.fft2((M - c1)**2)
  term3 = np.fft.fft2((M - c2)**2)

  term4 = np.fft.fft2(u)
  term5 = np.fft.fft2(B)

  #ipdb.set_trace()
  #numerator
  num = rho2*term1 - lamF*term2 + lamB*term3 + zeta5*term4 - eta*term5

  #denominator
  denom = zeta5 + rho2*reused['lap_kernel_f']

  u_new = np.real(np.fft.ifft2(num / denom))

  u_new = np.clip(u_new, 0, 1)
  return u_new

def region_subproblem_(params_dict, M, u, c1, c2):
  #solution of the proximal problems (40), (41) in the RKHS paper
  zeta3, zeta4, lamF, lamB = params_dict["zeta3"], params_dict["zeta4"], params_dict["lamF"], params_dict["lamB"]

  c1_new = (zeta3*c1 + 2*lamF*(u * M).sum()) / (zeta3 + 2*lamF*(u.sum()))
  out_u = 1 - u #region outside foreground
  c2_new = (zeta4*c2 + 2*lamB*(out_u * M).sum()) / (zeta4 + 2*lamB*(out_u.sum()))
  #placeholder
  return c1_new, c2_new


def g_f2(img1):
  out = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-12) #normalizing
  out2 = 1/(1 + (1e2)*(img1**2))
  return out2


def segment(params_dict, M, g, B, init_mask=None, verbose=False):
  iter_diagnostics = {}
  H, W = M.shape

  #finite difference matrix too large to compute
  #using single step forward differences
  # x - axis 1
  # y - axis 0
  m_ep = float(np.finfo(float).eps)


  #reused variables, like the fourier convolution kernel
  reused = create_reused_dict_(M)


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
    c1, c2 = region_subproblem_(params_dict, M, u, c1, c2)
    #u subproblem
    u = u_subproblem_(params_dict, reused, u, w_x, w_y, b2x, b2y, c1, c2, M, B, g)
    #w subproblem
    [w_x, w_y] = w_subproblem_(params_dict, reused, u, w_x, w_y, b2x, b2y, c1, c2, M, g)
    #b2 (split variable) update
    [b2x, b2y] = bregman_update_(u, w_x, w_y, b2x, b2y)

    if(verbose):
      print(f'iter: {i},  c1: {c1}  c2: {c2}')
  return u
