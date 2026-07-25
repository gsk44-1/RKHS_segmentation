from scipy.stats import norm
import numpy as np


NC = 4 #channels

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

def bregman_updates_(us, wxs, wys, b2xs, b2ys, b3, b4, b5, u_nuc, u_int, u_bdy, u_fg, v, o):
  #you should probably precompute this each outer iteration
  b2xs_new = np.zeros(b2xs.shape)
  b2ys_new = np.zeros(b2ys.shape)
  for j in range(0, NC):
    u_x = -us[j] + np.roll(us[j], -1, axis=1)
    u_y = -us[j] + np.roll(us[j], -1, axis=0)

    b2xs_new[j] = b2xs[j] + wxs[j] - u_x
    b2ys_new[j] = b2ys[j] + wys[j] - u_y

  b3 = b3 + u_nuc + v + o - (v*o) - u_fg
  b4 = b4 + u_int - v
  b5 = b5 + u_bdy - o

  return b2xs_new, b2ys_new, b3, b4, b5

def w_subproblems_(params_dict, reused, us, wxs, wys, b2xs, b2ys, gs):
  wxs_new, wys_new = np.zeros(wxs.shape), np.zeros(wys.shape)

  def shrink(x, t):
    return np.sign(x)*np.maximum(np.abs(x) - t, 0.0)

  for j in range(0, NC):
    mu = params_dict[j]['mu']
    rho2 = params_dict[j]['rho2']



    u_x = -us[j] + np.roll(us[j], -1, axis=1)
    u_y = -us[j] + np.roll(us[j], -1, axis=0)

    thresholds = mu/rho2 * gs[j]

    wxs_new[j] = shrink(u_x - b2xs[j], thresholds)
    wys_new[j] = shrink(u_y - b2ys[j], thresholds)

  return wxs_new, wys_new


def v_update(params_dict, reused, u_int, u_nuc, v, o, u_fg, b3, b4):
  #can be solved entrywise,
  rho3 = params_dict[0]['rho3']
  rho4 = params_dict[0]['rho4']#keep these the same across 0, 1, 2

  num = -rho3 * ((np.ones(o.shape) - o)) * (u_nuc + o - u_fg + b3) + rho4 * (u_int + b4)
  denom = rho3 * ((np.ones(o.shape) - o) ** 2) + rho4
 
  v_new = num / denom
  return v_new

def o_update(params_dict, reused, u_bdy, u_nuc, v, o, u_fg, b3, b5):
  #can be solved entrywise,
  rho3 = params_dict[0]['rho3']
  rho5 = params_dict[0]['rho5']#keep these the same across 0, 1, 2
  num = -rho3*((np.ones(v.shape) - v))*(u_nuc + v - u_fg + b3) + rho5*(u_bdy + b5)
  denom = rho3*((np.ones(v.shape) - v)**2) + rho5

  o_new = num / denom
  return o_new

#for u subproblems
def _data_terms(reused, M, c1, c2, wx, wy, b2x, b2y, u, lam_fg, lam_bg, rho2, zeta5):
  """Pieces shared by every u-subproblem, in Fourier domain."""
  term1 = (np.conj(reused['x_kern_fourier']) * np.fft.fft2(wx + b2x)
           + np.conj(reused['y_kern_fourier']) * np.fft.fft2(wy + b2y))
  term2 = np.fft.fft2((M - c1) ** 2)
  term3 = np.fft.fft2((M - c2) ** 2)
  term4 = np.fft.fft2(u)
  return rho2 * term1 - lam_fg * term2 + lam_bg * term3 + zeta5 * term4
 
def u_bdy_subproblem_(params_dict, reused, M, u, c1, c2, wx, wy, b2x, b2y, o, b5):
  p = params_dict[2]
  lam, fg_prop = p['lam'], p['fg_prop']
  rho2, rho5, zeta5 = p['rho2'], p['rho5'], p['zeta5']
  lam_fg, lam_bg = lam * fg_prop, lam * (1 - fg_prop)
 
  base = _data_terms(reused, M, c1, c2, wx, wy, b2x, b2y, u, lam_fg, lam_bg, rho2, zeta5)
  term5 = np.fft.fft2(b5 - o)
 
  num = base - rho5 * term5
  denom = zeta5 + rho5 + rho2 * reused['lap_kernel_f']
 
  return np.clip(np.real(np.fft.ifft2(num / denom)), 0, 1)
 
 
def u_int_subproblem_(params_dict, reused, M, u, c1, c2, wx, wy, b2x, b2y, v, b4):
  p = params_dict[1]
  lam, fg_prop = p['lam'], p['fg_prop']
  rho2, rho4, zeta5 = p['rho2'], p['rho4'], p['zeta5']
  lam_fg, lam_bg = lam * fg_prop, lam * (1 - fg_prop)
 
  base = _data_terms(reused, M, c1, c2, wx, wy, b2x, b2y, u, lam_fg, lam_bg, rho2, zeta5)
  term5 = np.fft.fft2(b4 - v)
 
  num = base - rho4 * term5
  denom = zeta5 + rho4 + rho2 * reused['lap_kernel_f']
 
  return np.clip(np.real(np.fft.ifft2(num / denom)), 0, 1)
 
 
def u_nuclear_subproblem_(params_dict, reused, M, curv_map, u, c1, c2,
                          wx, wy, b2x, b2y, v, o, u_fg, b3):
  p = params_dict[0]
  lam, fg_prop = p['lam'], p['fg_prop']
  rho2, rho3, zeta5, eta = p['rho2'], p['rho3'], p['zeta5'], p['eta']
  lam_fg, lam_bg = lam * fg_prop, lam * (1 - fg_prop)
 
  base = _data_terms(reused, M, c1, c2, wx, wy, b2x, b2y, u, lam_fg, lam_bg, rho2, zeta5)
 
  # dE/du_nuc  =>  +rho3 * (everything else in the residual)
  term5 = np.fft.fft2(v + o - (v * o) - u_fg + b3)
  term6 = np.fft.fft2(curv_map)
 
  num = base - rho3 * term5 - eta * term6
  denom = zeta5 + rho3 + rho2 * reused['lap_kernel_f']
 
  return np.clip(np.real(np.fft.ifft2(num / denom)), 0, 1)
 
 
def u_fg_subproblem_(params_dict, reused, M, u, c1, c2, wx, wy, b2x, b2y,
                     u_nuc, v, o, b3, fg_prior=None):
  p = params_dict[3]
  lam, fg_prop = p['lam'], p['fg_prop']
  rho2, rho3, zeta5 = p['rho2'], p['rho3'], p['zeta5']
  zeta_fg = p.get('zeta_fg', 0.0)          # optional anchor to a prior
  lam_fg, lam_bg = lam * fg_prop, lam * (1 - fg_prop)
 
  base = _data_terms(reused, M, c1, c2, wx, wy, b2x, b2y, u, lam_fg, lam_bg, rho2, zeta5)
 
  term5 = np.fft.fft2(u_nuc + v + o - (v * o) + b3)   # note: + rho3, not -
  num = base + rho3 * term5
 
  use_prior = (zeta_fg > 0.0 and fg_prior is not None)
  if use_prior:
      num = num + zeta_fg * np.fft.fft2(fg_prior)
  denom = zeta5 + (zeta_fg if use_prior else 0.0) + rho3 + rho2 * reused['lap_kernel_f']
 
  return np.clip(np.real(np.fft.ifft2(num / denom)), 0, 1)
 
 
def u_subproblems_(params_dict, reused, M_block, curv_map, us, c1s, c2s,
                   wxs, wys, b2xs, b2ys, v, o, b3, b4, b5, fg_prior=None):
  us_new = np.zeros(us.shape)
 
  us_new[0] = u_nuclear_subproblem_(params_dict, reused, M_block[0], curv_map,
                                    us[0], c1s[0], c2s[0], wxs[0], wys[0],
                                    b2xs[0], b2ys[0], v, o, us[3], b3)
  us_new[1] = u_int_subproblem_(params_dict, reused, M_block[1], us[1],
                                c1s[1], c2s[1], wxs[1], wys[1],
                                b2xs[1], b2ys[1], v, b4)
  us_new[2] = u_bdy_subproblem_(params_dict, reused, M_block[2], us[2],
                                c1s[2], c2s[2], wxs[2], wys[2],
                                b2xs[2], b2ys[2], o, b5)
  us_new[3] = u_fg_subproblem_(params_dict, reused, M_block[3], us[3],
                               c1s[3], c2s[3], wxs[3], wys[3],
                               b2xs[3], b2ys[3], us_new[0], v, o, b3,
                               fg_prior=fg_prior)
  return us_new

def region_subproblems_(params_dict, M, us, c1s, c2s):
  #solution of the proximal problems (40), (41) in the RKHS paper
  c1_new = np.zeros(c1s.shape)
  c2_new = np.zeros(c2s.shape)

  for j in range(0, NC):
    zeta3, zeta4, lam, fg_prop = params_dict[j]["zeta3"], params_dict[j]["zeta4"], params_dict[j]["lam"], params_dict[j]["fg_prop"]

    lam_fg = lam*fg_prop
    lam_bg = lam*(1-fg_prop)

    c1_new[j] = (zeta3*c1s[j] + 2*lam_fg*(us[j] * M[j]).sum()) / (zeta3 + 2*lam_fg*(us[j].sum()))
    out_u = 1 - us[j] #region outside foreground
    c2_new[j] = (zeta4*c2s[j] + 2*lam_bg*(out_u * M[j]).sum()) / (zeta4 + 2*lam_bg*(out_u.sum()))
  return c1_new, c2_new


def segment_imgs(params_dict, M_block, curv_map, g_block,
                 init_mask=None, fg_prior=None, verbose=False):
  """
  M_block   : (4, H, W)  -- nuc, int, bdy, fg  images / probability maps
  g_block   : (4, H, W)  -- per-channel TV edge weights
  init_mask : (4, H, W)  -- initial soft indicators
  fg_prior  : (H, W) or None -- only used if params_dict[3]['zeta_fg'] > 0
  """
  H, W = M_block.shape[1:3]
  m_ep = float(np.finfo(float).eps)
  reused = create_reused_dict_(M_block[0])
 
  us = init_mask.astype(float)
 
  c1s = np.zeros((NC,))
  c2s = np.zeros((NC,))
  wxs = np.zeros((NC, H, W))
  wys = np.zeros((NC, H, W))
 
  for j in range(0, NC):
    c1s[j] = (M_block[j] * us[j]).sum() / (us[j].sum() + m_ep)
    c2s[j] = (M_block[j] * (1 - us[j])).sum() / ((1 - us[j]).sum() + m_ep)
    wxs[j] = -us[j] + np.roll(us[j], -1, axis=1)
    wys[j] = -us[j] + np.roll(us[j], -1, axis=0)
 
  b2xs = np.zeros((NC, H, W))
  b2ys = np.zeros((NC, H, W))
  b3 = np.zeros((H, W))
  b4 = np.zeros((H, W))
  b5 = np.zeros((H, W))
 
  v = np.clip(M_block[1], 0.0, 1.0)
  o = np.clip(M_block[2], 0.0, 1.0)
 
  for i in range(params_dict[0]["iterations"]):
    c1s, c2s = region_subproblems_(params_dict, M_block, us, c1s, c2s)
    us = u_subproblems_(params_dict, reused, M_block, curv_map, us, c1s, c2s,
                        wxs, wys, b2xs, b2ys, v, o, b3, b4, b5,
                        fg_prior=fg_prior)
    wxs, wys = w_subproblems_(params_dict, reused, us, wxs, wys, b2xs, b2ys, g_block)
    v = np.clip(v_update(params_dict, reused, us[1], us[0], v, o, us[3], b3, b4), 0.0, 1.0)
    o = np.clip(o_update(params_dict, reused, us[2], us[0], v, o, us[3], b3, b5), 0.0, 1.0)
    b2xs, b2ys, b3, b4, b5 = bregman_updates_(us, wxs, wys, b2xs, b2ys,
                                              b3, b4, b5,
                                              us[0], us[1], us[2], us[3], v, o)
 
    if verbose:
      r3 = us[0] + v + o - v * o - us[3]
      print(f"it {i:3d}  |r3|={np.abs(r3).mean():.4e}  "
            f"|u_int-v|={np.abs(us[1]-v).mean():.4e}  "
            f"|u_bdy-o|={np.abs(us[2]-o).mean():.4e}  "
            f"fg mass={us[3].sum():.1f}")
 
  return us
