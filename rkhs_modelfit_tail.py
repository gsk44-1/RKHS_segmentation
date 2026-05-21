    if verbose:
        PtP_op = cache["PtP_op"]
        L_beta = cache["L_beta"]
        z2_eff = cache["zeta2_eff"]
        z2_usr = cfg["opt"]["zeta2_betaprox"]
        z2_saf = cfg["opt"].get("zeta2_betaprox_safety", 0.0)
        print("[stage1] ||Psi^T Psi||_op = %.3f, "
              "L_beta = (1+rho2)||Psi^T Psi|| = %.3f, "
              "zeta2_eff = %.3f "
              "(user zeta2 = %.3f, safety = %s)"
              % (PtP_op, L_beta, z2_eff, z2_usr, z2_saf))

    n_iter  = int(cfg["opt"]["maxiter"])
    history = []

    for it in range(n_iter):
        _patch_update(image, state, cache, cfg)
        _W_update(state, cache, cfg)
        _v_and_bregman_update(state, cache, cfg)

        Kd = state["img"]["Kd"]
        Pb = state["img"]["Pb"]
        M  = Kd + Pb
        residual = float(np.linalg.norm(image - M))

        history.append({
            "iter":     it,
            "Kd":       Kd.copy(),
            "Pb":       Pb.copy(),
            "M":        M.copy(),
            "residual": residual,
        })
        if verbose:
            beta_l1 = float(np.sum(np.abs(state["patch"]["beta"])))
            print("iter %3d/%d  residual=%.6f  |beta|_1=%.6f"
                  % (it + 1, n_iter, residual, beta_l1))

    return {
        "Kd":       state["img"]["Kd"],
        "Psi_beta": state["img"]["Pb"],
        "M":        state["img"]["Kd"] + state["img"]["Pb"],
        "d":        state["patch"]["d"],
        "beta":     state["patch"]["beta"],
        "cfg":      cfg,
        "history":  history,
        # Raw basis matrices and the d-update inverse, for users who want to
        # inspect conditioning / column-overlap diagnostics directly.
        "matrices": {
            "K":       cache["K"],        # (ps^2, ps^2)
            "Psi":     cache["Psi"],      # (ps^2, num_offsets * L)
            "A_d_inv": cache["A_d_inv"],  # (ps^2, ps^2)
            "KtPsi":   cache["KtPsi"],    # (ps^2, num_offsets * L)
        },
        "diagnostics": {
            "K_op":      cache["K_op"],
            "PtP_op":    cache["PtP_op"],
            "L_beta":    cache["L_beta"],
            "zeta2_eff": cache["zeta2_eff"],
        },
    }


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------

def _apply_overrides(cfg, overrides):
    """Route flat keyword overrides into the nested cfg dict."""
    sections = ("mdl", "bss", "ptc", "opt")
    for k, v in overrides.items():
        placed = False
        for sec in sections:
            if k in cfg.get(sec, {}):
                cfg[sec][k] = v
                placed = True
                break
        if not placed:
            raise KeyError("unknown override: %r" % k)
    return cfg
