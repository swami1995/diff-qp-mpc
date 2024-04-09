using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.activate(@__DIR__)
# Pkg.instantiate()

using Libdl
using LinearAlgebra
using Printf
import ForwardDiff as FD
import Random
using PyCall
using JLD2    

include(joinpath(@__DIR__, "simple_altro.jl"))

##

#---------------------THIS IS WHAT YOU NEED TO INPUT--------
function discrete_dynamics(p::NamedTuple, x, u, k)
    nq = Int(p.nx / 2)
    q_next = zeros(nq)
    qdot_next = zeros(nq)
    h = [p.dt]
    q = x[1:nq]
    qdot = x[nq+1:end]
    u = [u; zeros(nq - 1)]

    ccall(p.forward_dynamics, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
        q, qdot, u, h, q_next, qdot_next)
    return [q_next; qdot_next]
end
function discrete_jacobians(p::NamedTuple, x, u)
    nq = Int(p.nx / 2)
    h = [p.dt]
    q = x[1:nq]
    qdot = x[nq+1:end]
    u = [u; zeros(nq - 1)]
    q_jac_qout = zeros(nq, nq)
    q_jac_qdotout = zeros(nq, nq)
    q_jac_uout = zeros(nq, nq)
    qdot_jac_qout = zeros(nq, nq)
    qdot_jac_qdotout = zeros(nq, nq)
    qdot_jac_uout = zeros(nq, nq)

    ccall(p.forward_derivatives, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cdouble},
            Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
        q, qdot, u, h, q_jac_qout, q_jac_qdotout, q_jac_uout, qdot_jac_qout, qdot_jac_qdotout,
        qdot_jac_uout)
    # x_jac_x is [q_jac_qout q_jac_qdotout; qdot_jac_qout qdot_jac_qdotout]
    x_jac_x = zeros(p.nx, p.nx)
    x_jac_x[1:nq, 1:nq] = q_jac_qout
    x_jac_x[1:nq, nq+1:end] = q_jac_qdotout
    x_jac_x[nq+1:end, 1:nq] = qdot_jac_qout
    x_jac_x[nq+1:end, nq+1:end] = qdot_jac_qdotout
    # x_jac_u is [q_jac_uout; qdot_jac_uout]
    x_jac_u = zeros(p.nx, p.nu)
    x_jac_u[1:nq, :] = q_jac_uout[:, 1]
    x_jac_u[nq+1:end, :] = qdot_jac_uout[:, 1]
    return x_jac_x, x_jac_u
end
function ineq_con_x(p, x)
    [x - p.x_max; -x + p.x_min]
end
function ineq_con_u(p, u)
    [u - p.u_max; -u + p.u_min]
end
function ineq_con_u_jac(params, u)
    FD.jacobian(_u -> ineq_con_u(params, _u), u)
end
function ineq_con_x_jac(p, x)
    FD.jacobian(_x -> ineq_con_x(p, _x), x)
end

# here is the script to run trajopt
let
    CARTPOLE_PATH = joinpath(@__DIR__, "cartpole2l/")
    lib = dlopen(joinpath(CARTPOLE_PATH, "build/libdynamics.so"))
    forward_dynamics = dlsym(lib, :forward_dynamics)
    forward_derivatives = dlsym(lib, :forward_derivatives)
    nq = 3
    nx = nq * 2
    nu = 1
    dt = 0.03
    tf = 5.0
    t_vec = 0:dt:tf
    N = length(t_vec)

    # x0 = [0, 0.1, 0.1, 0.0]
    # xg = [0, 0., 0, 0.0]
    x0 = [0, π, 0, 0.0, 0, 0]
    xg = [0, 0, 0, 0, 0, 0.0]
    Xref = [deepcopy(xg) for i = 1:N]
    Uref = [zeros(nu) for i = 1:N-1]
    # Q = 1e1 * Diagonal([1, 10, 1, 1.0])
    Q = 1e1 * Diagonal([1, 10, 10, 1, 1, 1.0])
    R = 1e-1 * Diagonal([1.0])
    Qf = 10 * Q

    u_min = -100. * ones(nu)
    u_max = 100. * ones(nu)

    # state is x y v θ
    x_min = -200 * ones(nx)
    x_max = 200 * ones(nx)

    ncx = 2 * nx
    ncu = 2 * nu

    params = (
        nx=nx,
        nu=nu,
        ncx=ncx,
        ncu=ncu,
        N=N,
        Q=Q,
        R=R,
        Qf=Qf,
        u_min=u_min,
        u_max=u_max,
        x_min=x_min,
        x_max=x_max,
        Xref=Xref,
        Uref=Uref,
        dt=dt,
        forward_dynamics=forward_dynamics,
        forward_derivatives=forward_derivatives
    )

    # Test dynamics
    # x = [0.5, 0.5, 0.3, 0.7, 2.2, 1.0]
    # x = [0, 3.141592653589793, 0, 0.]
    # x = [0.2, 1.2, 4.2, 1.8]
    # u = [49.99999999327998]
    # xn = discrete_dynamics(params,x,u,1)
    # A = FD.finite_difference_jacobian(_x -> discrete_dynamics(params,_x,u,1),x)  # AUTODIFF DOESN'T WORK, USE FiniteDiff.jl
    # B = FD.finite_difference_jacobian(_u -> discrete_dynamics(params,x,_u,1),u)
    # A1, B1 = discrete_jacobians(params,x,u)
    # println("\nxn = $xn")
    # println("\nA = ")
    # display(A)
    # println("\nB = ")
    # display(B)
    # println("\neA1 = ")
    # display(norm(A-A1)/norm(A))
    # println("\neB1 = ")
    # display(norm(B-B1)/norm(B))

    X = [deepcopy(x0) for i = 1:N]
    U = [1 * randn(nu) for i = 1:N-1]
    # for i = 1:Int(N/2)
    #     U[i] .= u_min
    #     U[Int(N/2)-1+i] .= u_max
    # end

    Xn = deepcopy(X)
    Un = deepcopy(U)

    file = load("traj.jld2")["single_stored_object"]
    print(file)
    X = file[1]
    U = file[2]

    P = [zeros(nx, nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu, nx) for i = 1:N-1] # feedback gain
    Xhist = iLQR(params, X, U, P, p, K, d, Xn, Un; atol=1e-2, max_iters=2000, verbose=true, ρ=1e0, ϕ=10.0)

    # for i = 1:N-1
    #     X[i+1] = discrete_dynamics(params, X[i], Un[i], i)
    #     println(X[i])
    #     # println(Un[i])
    # end
    display(Xn)
    display(Un)
    
    # traj = (Xn, Un)
    # save_object("traj.jld2", traj)
    
    # np = pyimport("numpy")
    # X_np = np.asarray(X)
    # U_np = np.asarray(U)
    # np.savez("traj", X_np=X_np, U_np=U_np)
end