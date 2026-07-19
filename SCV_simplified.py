from scipy.stats import norm
import numpy as np


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

def bregman_updates_(us, wxs, wys, b2xs, b2ys, b3, b4, b5, u_nuc, u_int, u_bdy, v, o, fg):
  #you should probably precompute this each outer iteration
  b2xs_new = np.zeros(b2xs.shape)
  b2ys_new = np.zeros(b2ys.shape)
  for j in range(0, 3):
    u_x = -us[j] + np.roll(us[j], -1, axis=1)
    u_y = -us[j] + np.roll(us[j], -1, axis=0)

    b2xs_new[j] = b2xs[j] + wxs[j] - u_x
    b2ys_new[j] = b2ys[j] + wys[j] - u_y

  b3 = b3 + u_nuc + v + o - (v*o) - fg
  b4 = b4 + u_int - v
  b5 = b5 + u_bdy - o

  return b2xs_new, b2ys_new, b3, b4, b5

def w_subproblems_(params_dict, reused, us, wxs, wys, b2xs, b2ys, c1s, c2s, M, gs):
  wxs_new, wys_new = np.zeros(wxs.shape), np.zeros(wys.shape)

  def shrink(x, t):
    return np.sign(x)*np.maximum(np.abs(x) - t, 0.0)

  for j in range(0, 3):
    mu = params_dict[j]['mu']
    rho2 = params_dict[j]['rho2']



    u_x = -us[j] + np.roll(us[j], -1, axis=1)
    u_y = -us[j] + np.roll(us[j], -1, axis=0)

    thresholds = mu/rho2 * gs[j]

    lhsx = u_x - b2xs[j]
    lhsy = u_y - b2ys[j]

    wx_new = shrink(lhsx, thresholds)
    wy_new = shrink(lhsy, thresholds)


    num = np.sum((wxs[j]-u_x)**2) + np.sum((wys[j]-u_y)**2)
    den = np.sum(u_x**2) + np.sum(u_y**2) + 1e-12
    #print(num/den)
    wxs_new[j] = wx_new
    wys_new[j] = wy_new

  return wxs_new, wys_new


def v_update(params_dict, reused, u_int, u_nuc, v, o, fg, b3, b4):
  #can be solved entrywise,
  rho3 = params_dict[0]['rho3']
  rho4 = params_dict[0]['rho4']#keep these the same across 0, 1, 2

  #num = -rho3*((np.ones(v.shape) - v)**2)*(u_nuc + v - fg + b3) + rho5*(u_bdy + b5)
  num = -rho3*((np.ones(o.shape) - o))*(u_nuc + o - fg + b3) + rho4*(u_int + b4)
  denom = rho3*((np.ones(o.shape) - o)**2) + rho4

  v_new = num / denom
  return v_new

def o_update(params_dict, reused, u_bdy, u_nuc, v, o, fg, b3, b5):
  #can be solved entrywise,
  rho3 = params_dict[0]['rho3']
  rho5 = params_dict[0]['rho5']#keep these the same across 0, 1, 2
  num = -rho3*((np.ones(v.shape) - v))*(u_nuc + v - fg + b3) + rho5*(u_bdy + b5)
  denom = rho3*((np.ones(v.shape) - v)**2) + rho5

  o_new = num / denom
  return o_new



def u_bdy_subproblem_(params_dict, reused, M, u, c1, c2, wx, wy, b2x, b2y, v, o, fg, b3, b5):
  #u is u_bdy
  lam = params_dict[2]['lam']
  fg_prop = params_dict[2]['fg_prop']
  rho2 = params_dict[2]['rho2']
  rho5 = params_dict[2]['rho5']
  zeta5 = params_dict[2]['zeta5']
  lam_fg = lam*fg_prop
  lam_bg = lam*(1-fg_prop)

  term1_x = wx + b2x
  term1_y = wy + b2y

  term1 = np.conj(reused['x_kern_fourier']) * np.fft.fft2(term1_x) + np.conj(reused['y_kern_fourier']) * np.fft.fft2(term1_y)
  term2 = np.fft.fft2((M - c1)**2)
  term3 = np.fft.fft2((M - c2)**2)
  term4 = np.fft.fft2(u)
  term5 = np.fft.fft2(b5 - o)

  num = rho2*term1 - (lam_fg)*term2 + (lam_bg)*term3 + zeta5*term4 - rho5*term5
  denom = zeta5 + rho5 + rho2*reused['lap_kernel_f']

  u_new = np.real(np.fft.ifft2(num / denom))
  u_new = np.clip(u_new, 0, 1)

  return u_new

def u_int_subproblem_(params_dict, reused, M, u, c1, c2, wx, wy, b2x, b2y, v, o, fg, b4):
  #u is u_int
  lam = params_dict[1]['lam']
  fg_prop = params_dict[1]['fg_prop']
  rho2 = params_dict[1]['rho2']
  rho4 = params_dict[1]['rho4']
  zeta5 = params_dict[1]['zeta5']
  lam_fg = lam*fg_prop
  lam_bg = lam*(1-fg_prop)

  term1_x = wx + b2x
  term1_y = wy + b2y

  term1 = np.conj(reused['x_kern_fourier']) * np.fft.fft2(term1_x) + np.conj(reused['y_kern_fourier']) * np.fft.fft2(term1_y)
  term2 = np.fft.fft2((M - c1)**2)
  term3 = np.fft.fft2((M - c2)**2)
  term4 = np.fft.fft2(u)
  term5 = np.fft.fft2(b4 - v)

  num = rho2*term1 - (lam_fg)*term2 + (lam_bg)*term3 + zeta5*term4 - rho4*term5
  denom = zeta5 + rho4 + rho2*reused['lap_kernel_f']

  u_new = np.real(np.fft.ifft2(num / denom))
  u_new = np.clip(u_new, 0, 1)

  return u_new

def u_nuclear_subproblem_(params_dict, reused, M, curv_map, u, c1, c2, wx, wy, b2x, b2y, v, o, fg, b3):
  #all arguments are assumed to be only those for the nuclear indicator
  lam = params_dict[0]['lam']
  fg_prop = params_dict[0]['fg_prop']
  rho2 = params_dict[0]['rho2']
  rho3 = params_dict[0]['rho3']
  zeta5 = params_dict[0]['zeta5']
  eta = params_dict[0]['eta']
  #delta = params_dict['delta']
  lam_fg = lam*fg_prop
  lam_bg = lam*(1-fg_prop)

  term1_x = wx + b2x
  term1_y = wy + b2y

  term1 = np.conj(reused['x_kern_fourier']) * np.fft.fft2(term1_x) + np.conj(reused['y_kern_fourier']) * np.fft.fft2(term1_y)
  term2 = np.fft.fft2((M - c1)**2)
  term3 = np.fft.fft2((M - c2)**2)
  term4 = np.fft.fft2(u)
  term5 = np.fft.fft2((v + o - (v*o) - fg + b3))
  term6 = np.fft.fft2(curv_map)#curvature

  num = rho2*term1 - (lam_fg)*term2 + (lam_bg)*term3 + zeta5*term4 - rho3*term5 - eta*term6
  denom = zeta5 + rho3 + rho2*reused['lap_kernel_f']

  u_new = np.real(np.fft.ifft2(num / denom))
  u_new = np.clip(u_new, 0, 1)
  return u_new


def u_subproblems_(params_dict, reused, M_block, curv_map, us, c1s, c2s, wxs, wys, b2xs, b2ys, v, o, fg, b3, b4, b5):
  us_new = np.zeros(us.shape)
  us_new[0] = u_nuclear_subproblem_(params_dict, reused, M_block[0], curv_map, us[0], c1s[0], c2s[0], wxs[0], wys[0], b2xs[0], b2ys[0], v, o, fg, b3)
  us_new[1] = u_int_subproblem_(params_dict, reused, M_block[1], us[1], c1s[1], c2s[1], wxs[1], wys[1], b2xs[1], b2ys[1], v, o, fg, b4)
  us_new[2] = u_bdy_subproblem_(params_dict, reused, M_block[2], us[2], c1s[2], c2s[2], wxs[2], wys[2], b2xs[2], b2ys[2], v, o, fg, b3, b5)
  return us_new

def region_subproblems_(params_dict, M, us, c1s, c2s):
  #solution of the proximal problems (40), (41) in the RKHS paper
  c1_new = np.zeros(c1s.shape)
  c2_new = np.zeros(c2s.shape)

  for j in range(0, 3):
    zeta3, zeta4, lam, fg_prop = params_dict[j]["zeta3"], params_dict[j]["zeta4"], params_dict[j]["lam"], params_dict[j]["fg_prop"]

    lam_fg = lam*fg_prop
    lam_bg = lam*(1-fg_prop)

    c1_new[j] = (zeta3*c1s[j] + 2*lam_fg*(us[j] * M[j]).sum()) / (zeta3 + 2*lam_fg*(us[j].sum()))
    out_u = 1 - us[j] #region outside foreground
    c2_new[j] = (zeta4*c2s[j] + 2*lam_bg*(out_u * M[j]).sum()) / (zeta4 + 2*lam_bg*(out_u.sum()))
  return c1_new, c2_new


def segment_imgs(params_dict, M_block, curv_map, g_block, fg, init_mask=None, verbose=False):
  iter_diagnostics = {}
  H, W = M_block.shape[1:3]

  m_ep = float(np.finfo(float).eps)
  reused = create_reused_dict_(M_block[0])


  #initialize
  us = init_mask.astype(float)

  c1s = np.zeros((3))
  c2s = np.zeros((3))

  wxs = np.zeros((3, H, W))
  wys = np.zeros((3, H, W))

  for j in range(3):
    c1s[j] = (M_block[j] * us[j]).sum() / (us[j].sum() + m_ep)
    c2s[j] = (M_block[j] * (1 - us[j])).sum() / ((1 - us[j]).sum() + m_ep)

    wxs[j] = -us[j] + np.roll(us[j], -1, axis=1) #forward single step finite diff
    wys[j] = -us[j] + np.roll(us[j], -1, axis=0)


  b2xs = np.zeros((3, H, W)) #zero initialized
  b2ys = np.zeros((3, H, W))
  b3 = np.zeros((H,W))
  b4 = np.zeros((H,W))
  b5 = np.zeros((H,W))

  #o and v
  v = M_block[1]
  o = M_block[2]

  for i in range(params_dict[0]["iterations"]):
    c1s, c2s = region_subproblems_(params_dict, M_block, us, c1s, c2s)
    us = u_subproblems_(params_dict, reused, M_block, curv_map, us, c1s, c2s, wxs, wys, b2xs, b2ys, v, o, fg, b3, b4, b5)
    wxs, wys = w_subproblems_(params_dict, reused, us, wxs, wys, b2xs, b2ys, c1s, c2s, M_block, g_block)
    v = v_update(params_dict, reused, us[1], us[0], v, o, fg, b3, b4)
    o = o_update(params_dict, reused, us[2], us[0], v, o, fg, b3, b5)
    b2xs, b2ys, b3, b4, b5 = bregman_updates_(us, wxs, wys, b2xs, b2ys, b3, b4, b5, us[0], us[1], us[2], v, o, fg)
    #print(f'v: {np.sum(v)}  o: {np.sum(o)}  ')
  return us
