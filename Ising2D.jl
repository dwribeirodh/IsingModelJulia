# Author: Daniel Ribeiro (ribei040@umn.edu)

using Distributions
using Plots
using ProgressBars
using Elliptic
using HCubature: hcubature
using FiniteDifferences: central_fdm
using DelimitedFiles: readdlm, writedlm
using Dates: today
using LaTeXStrings
using DataStructures
using GilbertCurves

abstract type periodic end
abstract type fixed end

"""
Domain mapping methods:
    These methods map between local and global domains. This
    permits the the simulation to be performed with a vector
    of length L^2 instead of a square matrix with side length L
"""

function get_local_domain(k, L)
    """
    maps a global domain index k to a local domain index pair (i,j)
    """
    if k % L == 0
        i = k ÷ L
        j = L
    else
        i = (k ÷ L) + 1
        j = k % L
    end
    return i, j
end

function get_global_domain(i, j, L)
    """
    function to map index pair (i,j) to global index k.
    Used to map a matrix onto a vector
    """
    return j + (i - 1) * L
end

"""
Lattice generation
    These methods work in generating random
    or correlated lattices. They're used in the
    beggining of the simulation to initialize the system.
"""

function generate_lattice(L, is_random)
    """
    this function generate a square Ising lattice
    with side length L (number of spins is L^2).
    is_random determines if a lattice is completely correlated
    or completely random
    """
    is_random ? lattice = Int8.(rand([-1 1], L^2)) : lattice = Int8.(ones(L^2))
    return lattice
end

function reform_lattice(lattice, L)
    """
    reconstructs the lattice
    """
    sqrlat = zeros(Int8, (L,L))
    for idx = 1:L^2
        i, j = get_local_domain(idx, L)
        sqrlat[i, j] = lattice[idx]
    end
    return sqrlat
end

function get_hilbert_mask(sqrlat)
    return GilbertCurves.gilbertorder(sqrlat)
end

"""
Periodic BCS neighbor methods
    These methods will find the 4 nearest neighbors of a
    spin (top, bottom, left, and right). Periodic BCs are
    applied.
"""

function get_top(L, i, j, bc_type::Type{periodic})
    """
    returns the top neighbor index k given an index pair (i,j).
    Applies periodic boundary conditions
    """
    return get_global_domain(ifelse(i == 1, L, i - 1), j, L)
end

function get_bottom(L, i, j, bc_type::Type{periodic})
    """
    returns the bottom neighbor index k given an index pair (i,j)
    Applies periodic boundary conditions
    """
    return get_global_domain(ifelse(i == L, 1, i + 1), j, L)
end

function get_left(L, i, j, bc_type::Type{periodic})
    """
    returns the left neighbor index k given an index pair (i,j)
    applies periodic bcs
    """
    return get_global_domain(i, ifelse(j == 1, L, j - 1), L)
end

function get_right(L, i, j, bc_type::Type{periodic})
    """
    returns the right neighbor index k given an index pair (i,j)
    applies periodic bcs
    """
    get_global_domain(i, ifelse(j == L, 1, j + 1), L)
end

"""
Fixed boundary neighbor methods
    These methods will find the 4 nearest neighbors of a
    spin (top, bottom, left, and right). Fixed BCs are
    applied
"""

function get_top(L, i, j, bc_type::Type{fixed})
    """
    returns the top neighbor index k given an index pair (i,j)
    Applies fixed boundary conditions
    """
    i = ifelse(i == 1, 0, i - 1)
    (i == 0) ? (return 0) : (return get_global_domain(i, j, L))
end

function get_bottom(L, i, j, bc_type::Type{fixed})
    """
    returns the bottom neighbor index k given an index pair (i,j)
    applies fixed boundary conditions
    """
    i = ifelse(i == L, 0, i + 1)
    (i == 0) ? (return 0) : (return get_global_domain(i, j, L))
end

function get_left(L, i, j, bc_type::Type{fixed})
    """
    returns the left neighbor index k given an index pair (i,j)
    applies fixed boundary conditions
    """
    j = ifelse(j == 1, 0, j - 1)
    (j == 0) ? (return 0) : (return get_global_domain(i, j, L))
end

function get_right(L, i, j, bc_type::Type{fixed})
    """
    returns the right neighbor index k given an index pair (i,j)
    applies fixed boundary conditions
    """
    j = ifelse(j == L, 0, j + 1)
    (j == 0) ? (return 0) : (return get_global_domain(i, j, L))
end

function get_nn_idx(L, i, j, bc_type)
    """
    returns an array with the indices
    of the nearest neighbors of (i,j).
    applies both types of bcs
    """
    return [
        get_top(L, i, j, bc_type),
        get_right(L, i, j, bc_type),
        get_left(L, i, j, bc_type),
        get_bottom(L, i, j, bc_type),
    ]
end

function get_nn(lattice, L, i, j, bc_type::Type{periodic})
    """
    finds the spin values of the nearest neighbors
    of spin "angle". applies periodic Bcs
    """
    nnidx = get_nn_idx(L, i, j, periodic)
    return lattice[nnidx]
end

function get_nn(lattice, L, i, j,  bc_type::Type{fixed})
    """
    finds the spin values of the nearest neighbors
    of spin "angle". applies fixed Bcs
    """
    nnidx = get_nn_idx(L, i, j, fixed)
    nnidx = [val for val in nnidx if val != 0]
    return lattice[nnidx]
end

"""
Hamiltonian and magnetization
"""

function get_energy(lattice, L, bc_type)
    """
    computes the hamiltonian of the system
    """
    energy = 0.0
    for (idx, spin) in enumerate(lattice)
        i, j = get_local_domain(idx, L)
        nn = get_nn(lattice, L, i, j, bc_type)
        for nbr in nn
            energy += lattice[idx] * nbr
        end
    end
    return -energy/2
end

"""
Wolff simulation methods
    these methods are used in Wolff simulation
"""
function wolff_step(
    lattice,
    # spidx,
    L,
    # energy,
    β,
    bc_type
    )
    """
    performs one time step of the
    Wolff algorithm
    """
    # i,j = (rand(1:L), rand(1:L))
    # k = get_global_domain(i,j,L)
    k = rand(1:L^2)
    state = lattice[k]
    cluster_size = 0
    p = 1 - exp(-2*β)
    cluster_queue = Queue{Int}()
    enqueue!(cluster_queue, k)
    lattice[k] = -state
    while isempty(cluster_queue) == false
        idx = first(cluster_queue) #global domain
        i2, j2 = get_local_domain(idx, L)
        dequeue!(cluster_queue)
        cluster_size += 1
        # nbrs = get_nn_idx(n, n, i2, j2)
        nbrs = get_nn_idx(L, i2, j2, bc_type)
        for nnidx in nbrs
            if lattice[nnidx] == state
                randn = rand()
                if randn < p
                    enqueue!(cluster_queue, nnidx)
                    lattice[nnidx] = -state
                end
            end
        end
    end
    return lattice, cluster_size
end

function sweep_wolff(
    T,
    epoch,
    freq,
    L,
    niter,
    bc_type,
    configspath
    )
    """
    runs simulation using Wolff algorithm for one temperature value
    returns internal energy, heat capacity, and magnetization
    """
    β = 1.0 / T
     (
     β > 1 / (2 / log(1+sqrt(2))) ?
     lattice = generate_lattice(L, false) :
     lattice = generate_lattice(L, true)
     )

         cv = 0
         spins_flipped = 0
         E = []
         M = []
         n_config_save = 1
         time = 0
         while spins_flipped < epoch
         # while time < epoch
             lattice, cluster_size = wolff_step(lattice, L, β, bc_type)
             spins_flipped += cluster_size
             # println(time)
             # if spins_flipped > 0.95*epoch && (spins_flipped % freq == 0)
            if spins_flipped > 0.95*epoch
                time += 1
                if (time % freq == 0)
                    energy = get_energy(lattice, L, bc_type)
                    #println(spins_flipped)
                    mag = abs(sum(lattice))
                    push!(E, energy)
                    push!(M, mag)
                    if niter == 1 && n_config_save <= 50
                        save_configs(lattice, L, T, time, configspath)
                        n_config_save += 1
                    end
                end
            end
         end
         sigma2 = var(E)
         cv = β^2*sigma2 / L^2
         E = mean(E) / L^2
         M = mean(M) / L^2
         return E, cv, M
end

function wolff_simulation(
    T,
    epoch,
    freq,
    L,
    bc_type,
    configspath;
    niter = 5
    )
    """
    generates thermodynamic data for 2D Ising model.
    Uses wolff algorithm.
    """
    println("Running Wolff simulation...")
    Tlen = length(T)
    Energy = zeros(Tlen, niter)
    Cv = zeros(Tlen, niter)
    Mag = zeros(Tlen, niter)
    for jidx in ProgressBar(1:niter)
        for (iidx, temp) in ProgressBar(enumerate(T))
            Energy[iidx, jidx], Cv[iidx, jidx], Mag[iidx, jidx] = sweep_wolff(
                temp,
                epoch,
                freq,
                L,
                jidx,
                bc_type,
                configspath
            )
        end
    end
    sim_data = [mean(Energy, dims=2) mean(Cv, dims=2) mean(Mag, dims=2)]
    sigma =  [std(Energy, dims=2) std(Cv, dims=2) std(Mag, dims=2)]


    return [sim_data sigma]
end

"""
metropolis simulation methods
    these methods are used in the MCMC simulation
"""

function metropolis_step(
    lattice,
    spidx,
    L,
    energy,
    β,
    bc_type
    )
    """
    performs one time step of metropolis algorithm
    """
    iidx, jidx = get_local_domain(spidx, L)
    nn = get_nn(lattice, L, iidx, jidx, bc_type)
    h = sum(nn)
    ΔE = 2 * h * lattice[spidx]
    y = exp(-β * ΔE)
    if rand() < y
        lattice[spidx] = -lattice[spidx]
        energy += ΔE
    end
    return lattice, energy
end

function sweep_metropolis(
    T,
    epoch,
    freq,
    L,
    niter,
    bc_type,
    configspath
    )
    """
    given temperature T, runs a simulation using the Metropolis algorithm
    Params
        epoch: number of spin flips
        freq: frequency with which to output data
        L: lattice length
    Returns
        E: specific internal energy approximation
        Cv: specific approximation for heat capacity
    """
    β = 1.0 / T
     (
     β > 1 / (2 / log(1+sqrt(2))) ?
     lattice = generate_lattice(L, false) :
     lattice = generate_lattice(L, true)
     )

    energy = get_energy(lattice, L, bc_type)

    cv = 0.0
    Energy = []
    Mag = []

    time = 1
    spidx = 1
    nconfig = 1
    while time < epoch
        lattice, energy = metropolis_step(lattice, spidx, L, energy, β, bc_type)
        (spidx == L^2) ? spidx = 1 : spidx += 1
        time += 1
        if (time > 0.95 * epoch) && (time % freq == 0)
            push!(Energy, energy)
            mag = abs(sum(lattice))
            push!(Mag, mag)
            if niter == 1 && nconfig <=500
                save_configs(lattice, L, T, time, configspath)
                nconfig += 1
            end
        end
    end
    v = var(Energy)
    cv = β^2 * v / L^2
    Energy = mean(Energy) / L^2
    Mag = mean(Mag) / L^2
    return Energy, cv, Mag
end

function metropolis_simulation(
    T,
    epoch,
    freq,
    L,
    bc_type,
    configspath;
    niter = 5
    )
    """
    generates thermodynamic data for 1D xy using Metropolis.
    """

    println("Running Metropolis simulation...")
    Tlen = length(T)
    Energy = zeros(Tlen, niter)
    Cv = zeros(Tlen, niter)
    Mag = zeros(Tlen, niter)
    for jidx in ProgressBar(1:niter)
        for (iidx, temp) in ProgressBar(enumerate(T))
            Energy[iidx,jidx], Cv[iidx,jidx], Mag[iidx,jidx] = sweep_metropolis(
                temp,
                epoch,
                freq,
                L,
                jidx,
                bc_type,
                configspath
            )
        end
    end
    sim_data = [mean(Energy, dims=2) mean(Cv, dims=2) mean(Mag, dims=2)]
    sigma =  [std(Energy, dims=2) std(Cv, dims=2) std(Mag, dims=2)]
    return [sim_data sigma]
end

function save_configs(lattice, L, T, time, configspath)
    """
    saves configurations to txt file
    file name will be:
    2d_ising_config_temperature_spinsflipped.txt
    """
    sqrlat = reform_lattice(lattice, L)
    lattice = get_hilbert_mask(sqrlat)
    fname = "2d_ising_config_" * string(T) * "_" * string(time) *"_"* string(L) * ".txt"
    open(configspath * fname, "w") do io
        writedlm(io, lattice)
    end
end

function plot_data(
        sim_data,
        T_sim,
        sigma,
        exact_data,
        T_exact,
        epoch,
        L,
        plotspath
    )
    """
    Plots data, saves figures to plotspath
    """

    println("Plotting and saving figures to: " * plotspath)
    shading = 0.4

    e_plot = plot(
        T_exact,
        exact_data[:, 1],
        label = "exact",
        c = "maroon",
        minorticks=:false,
        tick_direction = :out,
        framestyle=:box,
        thickness_scaling=1.6,
        grid=:none,
        guidefontsize=12
    )
    scatter!(e_plot,
        T_sim,
        sim_data[:, 1],
        ribbon = sigma[:, 1],
        fillalpha = shading,
        xlabel=L"T",
        ylabel=L"U",
        minorticks=:false,
        tick_direction = :out,
        framestyle=:box,
        thickness_scaling=1.6,
        grid=:none,
        guidefontsize=12,
        legend = :bottomright,
        legendfontsize=6,
        label = "MCMC",
        c = "lightblue"
    )

    savefig(e_plot, plotspath * "2d_ising_energy_"*string(epoch)*"_"*string(L)*".png")

    cv_plot = plot(
        T_exact,
        exact_data[:, 2],
        label = "exact",
        c = "maroon",
        minorticks=:false,
        tick_direction = :out,
        framestyle=:box,
        thickness_scaling=1.6,
        grid=:none,
        guidefontsize=12
    )
    scatter!(cv_plot,
        T_sim,
        sim_data[:, 2],
        ribbon = sigma[:, 2],
        fillalpha = shading,
        xlabel=L"T",
        ylabel=L"C_{v}/N",
        minorticks=:false,
        tick_direction = :out,
        framestyle=:box,
        thickness_scaling=1.6,
        grid=:none,
        guidefontsize=12,
        legend = :topright,
        legendfontsize=6,
        label = "MCMC",
        c = "lightblue"
    )

    savefig(cv_plot, plotspath * "2d_ising_cv_" * string(epoch) * "_" * string(L) * ".png")

    mag_plot = plot(
        T_exact,
        exact_data[:,3],
        label = "exact",
        c = "maroon",
        minorticks=:false,
        tick_direction = :out,
        framestyle=:box,
        thickness_scaling=1.6,
        grid=:none,
        guidefontsize=12
    )

    scatter!(mag_plot,
        T_sim,
        sim_data[:, 3],
        ribbon = sigma[:, 3],
        fillalpha = shading,
        xlabel=L"T",
        ylabel=L"M",
        minorticks=:false,
        tick_direction = :out,
        framestyle=:box,
        thickness_scaling=1.6,
        grid=:none,
        guidefontsize=12,
        legend = :topright,
        legendfontsize=6,
        label = "MCMC",
        c = "lightblue"
    )

    savefig(mag_plot, plotspath * "2d_ising_mag_" * string(epoch) * "_" * string(L) * ".png")
    # return (e_plot, cv_plot, mag_plot)
end

"""
simulation setup functions
"""

function get_params()
    params = Dict()
    open("config.txt") do f
        while !eof(f)
            line = readline(f)
            if '=' in line
                line = split(line, "=")
                key = line[1]
                val = line[2]
                val = convert_type(val)
                params[key] = val
            end
        end
    end
    T_sim = params["T_sim_initial_value"]:params["T_sim_step_size"]:params["T_sim_final_value"]
    T_exact = params["T_exact_initial_value"]:params["T_exact_step_size"]:params["T_exact_final_value"]
    epoch = params["epoch"]
    freq = params["freq"]
    L = params["L"]
    bc_type = params["bc_type"]
    bc_type == "fixed" ? bc_type = fixed : bc_type = periodic
    ag = params["ag"]
    return (
        T_sim,
        T_exact,
        epoch,
        freq,
        L,
        bc_type,
        ag)
end

function is_bool(name::SubString{String})::Bool
    if name == "true" || name == "false"
        return true
    else
        return false
    end
end

function is_int(name::SubString{String})::Bool
    if '.' in name || 'e' in name || 'w' in name
        return false
    else
        return true
    end
end

function is_float(name::SubString{String})::Bool
    if '.' in name && name != "sweetsourcod.lempel_ziv"
        return true
    else
        return false
    end
end

function convert_type(name::SubString{String})
    if is_float(name)
        name = parse(Float64, name)
    elseif is_int(name)
        name = parse(Int64, name)
    elseif is_bool(name)
        name = parse(Bool, name)
    else
        name = convert(String, name)
    end
    return name
end

"""
exact thermodynamic properties
"""

function get_exact_magnetization(T)
    """
    computes exact magnetization of the system
    """
    β = 1.0/T
    if T < 2.0 / log(1.0 + sqrt(2.0))
        m = (1.0 - sinh(2 * β) ^ (-4)) ^ (1.0 / 8)
    else
        m = 0
    end
    return m
end

function get_exact_energy(T)
    """
    computes the exact internal energy of system
    http://www.lps.ens.fr/~krzakala/ISINGMODEL.pdf
    """
    β = 1.0/T
    m = 2 * sinh(2 * β) / cosh(2 * β)^2
    u =  - 1 / tanh(2 * β) * (1.0 + (2 * tanh(2 * β) ^ 2 - 1.0) * (2.0 / pi) * K(m ^ 2))
    return u
end

function get_exact_free_energy(T)
    """
    computes the exact free energy of the system
    http://www.lps.ens.fr/~krzakala/ISINGMODEL.pdf
    """
    β = 1.0/T
    integral = hcubature(t -> log(cosh(2 * β) ^ 2 - sinh(2 * β) * (cos(t[1]) + cos(t[2]))),
        [0,0], [pi,pi])[1]
    f = (-1.0 / β) * (log(2) + integral / (2 * pi ^ 2))
    return f
end

function get_exact_entropy(T)
    """
    computes exact entropy of the system
    """
    s = (get_exact_energy(T) - get_exact_free_energy(T)) / T
    return s / log(2)
end

function get_exact_cv(T)
    """
    approximates heat capacity with centered finite difference
    """
    cv = T * central_fdm(10, 1)(get_exact_entropy, T)
    return cv
end

function get_exact_properties(T)
    """
    wrapper function to compute exact thermodynamic quantities given
    a temperature range T
    """
    Energy = zeros(length(T))
    cv = zeros(length(T))
    Mag = zeros(length(T))
    println("Calculating exact results")
    for (i,temp) in ProgressBar(enumerate(T))
        cv[i] = get_exact_cv(temp)
        Energy[i] = get_exact_energy(temp)
        Mag[i] = get_exact_magnetization(temp)
    end
    return [Energy cv Mag]
end

function make_gif_mag(
    T,
    epoch,
    freq,
    L,
    bc_type,
    FPS
    )
    """
    given temperature T, runs a simulation using the Metropolis algorithm
    Params
        epoch: number of spin flips
        freq: frequency with which to output data
        L: lattice length
    Returns
        E: specific internal energy approximation
        Cv: specific approximation for heat capacity
    """
    β = 1.0 / T
     (
     β < 0.4407 ?
     lattice = generate_lattice(L, false) :
     lattice = generate_lattice(L, true)
     )
    energy = get_energy(lattice, L, bc_type)
    Time = []
    # Energy = []
    Mag = []
    # cv = 0.0

    time = 1
    # spidx = 1
    spin_flips = 0
    a = Animation()
    # e_exact = get_exact_energy(T)
    mag_exact = get_exact_magnetization(T)
    while time < epoch
        # (spidx == L^2) ? spidx = 1 : spidx += 1
        # lattice, energy, flip = metropolis_step(lattice, spidx, L, energy, β, bc_type)
        lattice, cluster_size = wolff_step(lattice, L, β, bc_type)
        spin_flips += cluster_size
        mag = abs(sum(lattice) / L^2)
        push!(Time, time)
        push!(Mag, mag)
        time += 1
        if (time % freq == 0)
            println(time/epoch*100)
            rl = reform_lattice(lattice,L)
            hmap = heatmap(rl, title = string(round(time/epoch*100, digits = 2))*"%", legend = :none, aspect_ratio = :equal)
            eplot = plot(Time, Mag, label = "MCMC", title = string(spin_flips))
            plot!(eplot, [0;time], [mag_exact; mag_exact], label = "exact value (T="*string(T)*")", legend = :none)
            xlabel!(eplot, "time")
            ylabel!(eplot, L"M/N")
            p = plot(hmap, eplot, layout = (1, 2))
            frame(a, p)
        end
    end
    # v = var(Energy)
    # cv = β^2 * v / L^2
    # Energy = mean(Energy) / L^2
    gif(a, "2D_Ising_Magnetization.gif", fps = FPS)
    return [Time Mag]
end

function make_gif_u(
    T,
    epoch,
    freq,
    L,
    bc_type,
    FPS
    )
    """
    given temperature T, runs a simulation using the Metropolis algorithm
    Params
        epoch: number of spin flips
        freq: frequency with which to output data
        L: lattice length
    Returns
        E: specific internal energy approximation
        Cv: specific approximation for heat capacity
    """
    β = 1.0 / T
     (
     β < 0.4407 ?
     lattice = generate_lattice(L, false) :
     lattice = generate_lattice(L, true)
     )
    energy = get_energy(lattice, L, bc_type)
    Time = []
    Energy = []
    Mag = []
    # cv = 0.0

    time = 1
    # spidx = 1
    spin_flips = 0
    a = Animation()
    # e_exact = get_exact_energy(T)
    energy_exact = get_exact_energy(T)
    while spin_flips < epoch
        # (spidx == L^2) ? spidx = 1 : spidx += 1
        # lattice, energy, flip = metropolis_step(lattice, spidx, L, energy, β, bc_type)
        lattice, cluster_size = wolff_step(lattice, L, β, bc_type)
        spin_flips += cluster_size
        energy = get_energy(lattice, L, bc_type) / L^2
        push!(Time, time)
        push!(Energy, energy)
        time += 1
        if (time % freq == 0)
            println(time/epoch*100)
            rl = reform_lattice(lattice,L)
            hmap = heatmap(rl, title = string(round(time/epoch*100, digits = 2))*"%", legend = :none, aspect_ratio = :equal)
            eplot = plot(Time, Energy, label = "MCMC", title = string(spin_flips))
            plot!(eplot, [0;time], [energy_exact; energy_exact], label = "exact value (T="*string(T)*")", legend = :none)
            xlabel!(eplot, "time")
            ylabel!(eplot, L"M/N")
            p = plot(hmap, eplot, layout = (1, 2))
            frame(a, p)
        end
    end
    # v = var(Energy)
    # cv = β^2 * v / L^2
    # Energy = mean(Energy) / L^2
    gif(a, "2D_Ising_Energy.gif", fps = FPS)
    return [Time Energy]
end

function make_gif_cv(
    T,
    epoch,
    freq,
    L,
    bc_type,
    FPS
    )
    """
    given temperature T, runs a simulation using the Metropolis algorithm
    Params
        epoch: number of spin flips
        freq: frequency with which to output data
        L: lattice length
    Returns
        E: specific internal energy approximation
        Cv: specific approximation for heat capacity
    """
    β = 1.0 / T
     (
     β < 0.4407 ?
     lattice = generate_lattice(L, false) :
     lattice = generate_lattice(L, true)
     )
    Time = []
    Energy = []
    Cv = []
    # cv = 0.0

    time = 1
    # spidx = 1
    spin_flips = 0
    a = Animation()
    # e_exact = get_exact_energy(T)
    cv_exact = get_exact_cv(T)
    while time < epoch
        # (spidx == L^2) ? spidx = 1 : spidx += 1
        # lattice, energy, flip = metropolis_step(lattice, spidx, L, energy, β, bc_type)
        lattice, cluster_size = wolff_step(lattice, L, β, bc_type)
        spin_flips += cluster_size
        energy = get_energy(lattice, L, bc_type)
        push!(Energy, energy)
        time += 1
        if (time % freq == 0)
            cv = β^2 * var(Energy) / L^2
            push!(Time, time)
            push!(Cv, cv)
            println(time/epoch*100)
            rl = reform_lattice(lattice,L)
            hmap = heatmap(rl, title = string(round(time/epoch*100, digits = 2))*"%", legend = :none, aspect_ratio = :equal)
            eplot = plot(Time, Cv, label = "MCMC", title = string(spin_flips))
            plot!(eplot, [0;time], [cv_exact; cv_exact], label = "exact value (T="*string(T)*")", legend = :none)
            xlabel!(eplot, "time")
            ylabel!(eplot, L"M/N")
            p = plot(hmap, eplot, layout = (1, 2))
            frame(a, p)
            E = []
        end
    end
    # v = var(Energy)
    # cv = β^2 * v / L^2
    # Energy = mean(Energy) / L^2
    gif(a, "2D_Ising_Cv.gif", fps = FPS)
    return [Time Cv]
end

function get_wolff_steps2eq_u(
    T,
    freq,
    L,
    rtol,
    bc_type
    )
    """
    runs simulation using Wolff algorithm for one temperature value
    returns internal energy, heat capacity, and magnetization
    """
    exact_u = get_exact_energy(T) # benchmark
    β = 1.0 / T
    (
    β > 0.4407 ?
    lattice = generate_lattice(L, false) :
    lattice = generate_lattice(L, true)
    )
    # create arrays where data is stored
    # initialize vars
    expt_u = 0
    spins_flipped = 0.0
    time = 1
    # run simulation until magnetization converges
    while abs((exact_u - expt_u) / exact_u) >= rtol
        lattice, cluster_size = wolff_step(lattice, L, β, bc_type)
        spins_flipped += cluster_size
        if (time % freq == 0)
            # println(abs((exact_u - expt_u) / exact_u))
            expt_u = get_energy(lattice, L, bc_type) / L^2
        end
        time += 1
    end
    return spins_flipped
end

function get_wolff_steps2eq_cv(
    T,
    freq,
    L,
    rtol,
    bc_type
    )
    """
    runs simulation using Wolff algorithm for one temperature value
    returns internal energy, heat capacity, and magnetization
    """
    exact_cv = get_exact_cv(T) # benchmark
    β = 1.0 / T
    (
    β > 0.4407 ?
    lattice = generate_lattice(L, false) :
    lattice = generate_lattice(L, true)
    )
    # create arrays where data is stored
    # initialize vars
    expt_cv = 0.0
    spins_flipped = 0
    time = 1
    Energy = []
    # run simulation until cv converges
    while abs((exact_cv - expt_cv) / exact_cv) >= rtol
        lattice, cluster_size = wolff_step(lattice, L, β, bc_type)
        spins_flipped += cluster_size
        u = get_energy(lattice, L, bc_type)
        push!(Energy, u)
        if time % 1000 == 0
            println(abs((exact_cv - expt_cv) / exact_cv))
            expt_cv = β^2*var(Energy) / L^2
            E = []
        end
        time += 1
    end
    return spins_flipped
end

function get_wolff_steps2eq_mag(
    T,
    freq,
    L,
    rtol,
    bc_type
    )
    """
    runs simulation using Wolff algorithm for one temperature value
    returns internal energy, heat capacity, and magnetization
    """
    exact_mag = get_exact_magnetization(T) # benchmark
    β = 1.0 / T
    (
    β > 0.4407 ?
    lattice = generate_lattice(L, false) :
    lattice = generate_lattice(L, true)
    )
    # create arrays where data is stored
    # initialize vars
    expt_mag = 0
    spins_flipped = 0.0
    time = 1
    # run simulation until magnetization converges
    while abs((exact_mag - expt_mag) / exact_mag) >= rtol
        lattice, cluster_size = wolff_step(lattice, L, β, bc_type)
        spins_flipped += cluster_size
        if (time % freq == 0)
            expt_mag = abs(sum(lattice) / L^2)
        end
        time += 1
    end
    return spins_flipped
end

function get_metropolis_steps2eq(
    T,
    freq,
    L,
    rtol,
    bc_type
    )
    """
    runs simulation using Wolff algorithm for one temperature value
    returns internal energy, heat capacity, and magnetization
    """
    exact_cv = get_exact_cv(T) # benchmark

    β = 1.0 / T
    (
    β > 0.4407 ?
    lattice = generate_lattice(L, false) :
    lattice = generate_lattice(L, true)
    )

    lattice = generate_lattice(L, true)
    energy = get_energy(lattice, L, bc_type)

    time = 1
    spidx = 1
    Energy = zeros(freq)
    expt_cv = 1e5
    while abs((exact_cv - expt_cv) / exact_cv) >= rtol
        lattice, energy = metropolis_step(lattice, spidx, L, energy, β, bc_type)
        Energy[time] = energy
        if time % freq == 0
            expt_cv = β^2 * var(Energy) / L^2
            Energy = zeros(freq)
        end
        (spidx == L^2) ? spidx = 1 : spidx += 1
        time += 1
    end
    return time
end

function wolff_steps2eq_u(
    T,
    freq,
    L,
    rtol,
    bc_type
    )
    p = plot()
    for (jidx, len) in enumerate(L)
        println("Equilibrating lattice of length = "*string(len))
        data = zeros(length(T), 2)
        for (iidx, temp) in ProgressBar(enumerate(T))
            spin_flips = get_wolff_steps2eq_u(temp,freq,len,rtol,bc_type)
            data[iidx, 1] = temp
            data[iidx, 2] = spin_flips
        end
        plot!(p, data[:,1], data[:,2], label = string(len), title = title = "U (rtol = "*string(round(rtol, digits=4))*")")
    end
    xlabel!(L"T")
    ylabel!(L"N_{\text{flip}}")
    savefig(p, "wolff_time2eq_u.png")
end

function wolff_steps2eq_cv(
    T,
    freq,
    L,
    rtol,
    bc_type
    )
    p = plot()
    for (jidx, len) in enumerate(L)
        println("Equilibrating lattice of length = "*string(len))
        data = zeros(length(T), 2)
        for (iidx, temp) in ProgressBar(enumerate(T))
            spin_flips = get_wolff_steps2eq_cv(temp,freq,len,rtol,bc_type)
            data[iidx, 1] = temp
            data[iidx, 2] = spin_flips
        end
        plot!(p, data[:,1], data[:,2], label = string(len), title = "Cv (rtol = "*string(rtol)*")")
    end
    xlabel!(L"T")
    ylabel!(L"N_{\text{flip}}")
    savefig(p, "wolff_time2eq_cv.png")
end

function wolff_steps2eq_mag(
    T,
    freq,
    L,
    rtol,
    bc_type
    )
    p = plot()
    for (jidx, len) in enumerate(L)
        println("Equilibrating lattice of length = "*string(len))
        data = zeros(length(T), 2)
        for (iidx, temp) in ProgressBar(enumerate(T))
            spin_flips = get_wolff_steps2eq_mag(temp,freq,len,rtol,bc_type)
            data[iidx, 1] = temp
            data[iidx, 2] = spin_flips
        end
        plot!(p, data[:,1], data[:,2], label = string(len), title = "M (rtol = "*string(round(rtol, digits=4))*")")
    end
    xlabel!(L"T")
    ylabel!(L"N_{\text{flip}}")
    savefig(p, "wolff_time2eq_mag.png")
end

function wolff_steps2eq(
    T,
    freq,
    L,
    rtol,
    bc_type,
    thermo_quant
    )
    if thermo_quant == "internal energy"
        wolff_steps2eq_u(T, freq, L, rtol, bc_type)
    elseif thermo_quant == "heat capacity"
        wolff_steps2eq_cv(T, freq, L, rtol, bc_type)
    else
        wolff_steps2eq_mag(T, freq, L, rtol, bc_type)
    end
end

function metropolis_steps2eq(
    T,
    freq,
    L,
    rtol,
    bc_type
    )
    p = plot()
    for len in L
        println("Equilibrating lattice of length = "*string(len))
        data = zeros(length(T), 2)
        for (iidx, temp) in ProgressBar(enumerate(T))
            spin_flips = get_metropolis_steps2eq(temp,freq,len,rtol,bc_type)
            data[iidx, 1] = temp
            data[iidx, 2] = spin_flips
        end
        plot!(p, data[:,1], data[:,2], label = string(len), yaxis = :log)
    end
    xlabel!(L"T")
    ylabel!(L"N_{flip}")
    savefig(p, "metropolis_time2eq.png")
end

function plot_gif(rl, time, epoch, T, Time, u, u_exact)

    hmap = heatmap(rl,
    title = string(round(time/epoch*100, digits = 2))*"%"*", T = "*string(T),
    legend = :none,
    minorticks=:false,
    tick_direction = :out,
    framestyle=:box,
    aspect_ratio = :equal,
    grid=:none,
    )

    figplot1 = plot(Time,
    u,
    xlabel = L"N_f",
    ylabel = L"U / N",
    label = "MCMC",
    minorticks=:false,
    tick_direction = :out,
    framestyle=:box,
    grid=:none,
    legend = :best,
    c = "maroon"
    )
    plot!(figplot1, [0;time], [u_exact;u_exact], label = "exact value (T="*string(T)*")", c = "black")

    p = plot(hmap, figplot1, layout = (1,2))
    return p
end

function make_gif_u_metropolis(
    T,
    epoch,
    freq,
    L,
    bc_type,
    FPS,
    gifpath
    )

    β = 1.0 / T
    lattice = generate_lattice(L, true)

    energy = get_energy(lattice, L, bc_type)

    cv = 0.0
    Time = []
    Energy = []
    Mag = []

    time = 1
    spidx = 1
    a = Animation()
    a2 = Animation()
    energy_exact = get_exact_energy(T)
    mag_exact = get_exact_magnetization(T)
    p = 1
    while time < epoch
        lattice, energy = metropolis_step(lattice, spidx, L, energy, β, bc_type)
        push!(Time, time)
        push!(Energy, energy / L^2)
        push!(Mag, abs(sum(lattice)) / L^2)
        (spidx == L^2) ? spidx = 1 : spidx += 1
        time += 1
        if (time % freq == 0)
            println(time/epoch*100)
            rl = reform_lattice(lattice,L)
            p = plot_gif(rl, time, epoch, T, Time, Energy, energy_exact)
            p2 = plot_gif(rl, time, epoch, T, Time, Mag, mag_exact)
            frame(a, p)
            frame(a2, p2)
        end
    end
    gif(a, gifpath*"gif_metropolis_U_epoch_"*string(epoch)*".gif", fps = FPS)
    gif(a2, gifpath*"gif_metropolis_Mag_epoch_"*string(epoch)*".gif", fps = FPS)
end

function make_gif_u_wolff(
    T,
    epoch,
    freq,
    L,
    bc_type,
    FPS,
    gifpath
    )

    β = 1.0 / T
    lattice = generate_lattice(L, true)

    energy = get_energy(lattice, L, bc_type)

    cv = 0.0
    Time = []
    Energy = []
    Mag = []

    time = 1
    a = Animation()
    a2 = Animation()
    energy_exact = get_exact_energy(T)
    mag_exact = get_exact_magnetization(T)
    p = 1
    spin_flips = 0
    while spin_flips < epoch
        println(string(spin_flips/epoch*100))
        lattice, cluster_size = wolff_step(lattice, L, β, bc_type)
        spin_flips += cluster_size
        energy = get_energy(lattice, L, bc_type) / L^2
        push!(Time, spin_flips)
        push!(Energy, energy)
        push!(Mag, abs(sum(lattice)) / L^2)
        time += 1
        if (time % freq == 0)
            println(time/epoch*100)
            rl = reform_lattice(lattice,L)
            p = plot_gif(rl, spin_flips, epoch, T, Time, Energy, energy_exact)
            p2 = plot_gif(rl, spin_flips, epoch, T, Time, Mag, mag_exact)
            frame(a, p)
            frame(a2, p2)
        end
    end
    gif(a, gifpath*"gif_wolff_U_epoch_"*string(epoch)*".gif", fps = FPS)
    gif(a2, gifpath*"gif_wolff_Mag_epoch_"*string(epoch)*".gif", fps = FPS)

    # v = var(Energy)
    # cv = β^2 * v / L^2
    # Energy = mean(Energy) / L^2
    # Mag = mean(Mag) / L^2
    # return Energy, cv, Mag
    # return p
end

"""
main method
"""

function main()
    T_sim, T_exact, epoch, freq, L, bc_type, ag = get_params()
    ising_repo_path = pwd()
    today_date = string(today())
    mkpath("Simulation_Results/"*today_date*"/configs/")
    mkpath("Simulation_Results/"*today_date*"/plots/wolff/")
    mkpath("Simulation_Results/"*today_date*"/plots/metropolis/")
    mkpath("Simulation_Results/"*today_date*"/gifs/")
    # mkpath("Simulation_Results/"*today_date*"/energy/") # not currently being  used
    configs_path = ising_repo_path*"/Simulation_Results/"*today_date*"/configs/"
    plots_path = ising_repo_path*"/Simulation_Results/"*today_date*"/plots/"
    gif_path = ising_repo_path*"/Simulation_Results/"*today_date*"/gifs/"
    exact_data = get_exact_properties(T_exact)
    # if ag == "metropolis"
        metro_data = metropolis_simulation(T_sim, epoch, freq, L, bc_type, configs_path)
        plot_data(metro_data[:,1:3],
        T_sim,
        metro_data[:,4:end],
        exact_data,
        T_exact,
        epoch,
        L,
        plots_path*"metropolis/"
        )
    # else
    # wolff_data = wolff_simulation(T_sim, epoch, freq, L, bc_type, configs_path)
    # plot_data(wolff_data[:, 1:3],
    # T_sim,
    # wolff_data[:, 4:end],
    # exact_data,
    # T_exact,
    # epoch,
    # L,
    # plots_path*"wolff/"
    # )
    # return wolff_data
    # return metro_data
    # return (p1,p2,p3)
    println("##### End of simulation #####")
end

cd("/home/mart5523/ribei040/IsingModelJulia")
main()


# include("2DEntropy.jl")
# plot_data(wolff_data = wolff_simulation(T_sim, epoch, freq, L, bc_type, configs_path)

# temp = 3.0
# epoch = 10000
# freq = 1
# L = 100
# FPS = 10
# gif_path = pwd()
# make_gif_u_wolff(temp, epoch, freq, L, periodic, FPS, gif_path)
# make_gif_u_metropolis(temp, epoch, freq, L, periodic, FPS, gif_path)


# In order to check how long it would take a lattice of side length L
# being equilibrated over temperature range T, you can run the command below
# It will compute the exact value of the internal energy and then run the simulation
# until the criterion abs(exact_energy - simulation_energy) <= rtol holds true.
# It will also save a plot with the data
# the parameters of both function are:
#   -- temperature range
#   -- frequency with which to compute energy
#   -- vector or scalar of lattice lengths to be sampled
#   -- tolerance of convergence
#   -- type of boundary condition
#
# For wolff:
# wolff_steps2eq(0.5:1.0:5, 1000, [100], 10^-1, periodic, "heat capacity")
# wolff_steps2eq(0.5:0.1:5, 1000, [100 200 300], 10^-3, periodic, "internal energy")
# wolff_steps2eq(0.5:0.1:5, 1000, [100 200 300], 10^-3, periodic, "magnetization")
#
# For metropolis:
# metropolis_steps2eq(1.0:0.5:5, 1000, [100 200 500], 10^-2, periodic)
