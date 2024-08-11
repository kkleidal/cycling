import jax.numpy as jnp

def f_gravity(grade, weight_kg):
    return 9.8067 * jnp.sin(jnp.arctan(grade / 100)) * weight_kg

def f_rolling(grade, weight_kg, c_rr):
    return 9.8067 * jnp.cos(jnp.arctan(grade / 100)) * weight_kg * c_rr

def f_drag(v_headwind, v_groundspeed, cda, rho):
    v_airspeed = v_headwind + v_groundspeed
    return 0.5 * cda * rho * v_airspeed**2

def power_required(grade, weight_kg, c_rr, v_headwind, v_groundspeed, cda, rho, loss_drivetrain):
    return (1 - loss_drivetrain / 100)**(-1) * (
        f_gravity(grade=grade, weight_kg=weight_kg)
        + f_rolling(grade=grade, weight_kg=weight_kg, c_rr=c_rr)
        + f_drag(v_headwind=v_headwind, v_groundspeed=v_groundspeed, cda=cda, rho=rho)
    ) * v_groundspeed