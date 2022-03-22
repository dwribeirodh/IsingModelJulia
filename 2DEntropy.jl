using DelimitedFiles
using ProgressBars
using Plots
using PyCall
using HCubature: hcubature
using Elliptic
using LaTeXStrings
using Distributions
using Dates: today
using Distributions

function read_config_file(fname, path)
    """
    function to read the text files. returns array with temperatures
    and array of configurations. make sure fname includes the path to
    the configurations
    """
    idx = split(fname, "_")
    temp = parse(Float64, idx[4])
    config = readdlm(path*fname, ',')
    return config, temp
end

function get_stats(h_data_fname, h_path, ntemp, nconfig)
    h_data = readdlm(h_path*h_data_fname, ',')
    final_data = zeros(ntemp, 2)
    println("----------")
    println("Calculating stats for entropy values...")
    for (iidx) in ProgressBar(1:ntemp)
        h = zeros(nconfig)
        for (jidx) in 1:nconfig
            row = iidx*jidx
            h[jidx] = h_data[row]
        end
        final_data[iidx,1] = mean(h)
        final_data[iidx,2] = std(h)
    end
    return final_data
end

function dir_parser(L, path, sc, nT, nit)
    """
    parse directory of config data and return vector of entropy data
    and vector of temperature data
    """
    println("Parsing directory...")
    dir_list = readdir(path)
    temp_data = []
    h_data = []
    h_data2 = []
    d = Bernoulli(0.5)
    # cid_rand = get_cid_rand(L, 1000, sc)
    for fname in ProgressBar(dir_list)
        if fname != ".DS_Store"
            lattice, temp = read_config_file(fname, path)
            latice = shift_vec_vals(lattice)
            h = get_entropy_lz77(lattice, L, d, sc)
            h1 = h[1]
            h2 = h[2]
            # h = sc.lempel_ziv_complexity(x, "lz77")[2] / L
            push!(temp_data, temp)
            push!(h_data, h1)
            push!(h_data2, h2)
        end
    end
    data = [temp_data h_data h_data2]
    data = data[sortperm(data[:, 1]), :]
    data = data[sortperm(data[:, 1]), :]
    final_data = zeros(nT, 5)
    for i = 1:nT
        h1 = []
        h2 = []
        tc = 0
        for j = 1:nit
            row = i*j
            push!(h1, data[row,2])
            push!(h2, data[row,3])
            if j == nit
                final_data[i,1] = data[row,1]
            end
        end
        final_data[i,2] = mean(h1)
        final_data[i,3] = std(h1)
        final_data[i,4] = mean(h2)
        final_data[i,5] = std(h2)
    end
    return final_data, data
    # return data
end

function shift_vec_vals(lattice)
    for (idx,val) in enumerate(lattice)
        if val == -1
            lattice[idx] = 0
        end
    end
    return lattice
end

function get_entropy(x, L, cid_rand, sc)
    # L = length(x)
    # cid_rand = get_cid_rand(L, niter, sc)
    cid = sc.lempel_ziv_complexity(x, "lz77")[2] / L
    s = cid / cid_rand
    return s
end

function get_cid_rand(L, niter, sc)
    """
    computes the CID of random binary sequence over niter
    iterations for statistical accuracy.
    """
    cid_rand = 0.0
    d = Bernoulli(0.5)
    println("Computing cid_rand ...")
    for i in ProgressBar(1:niter)
        rand_seq = rand(d, L)
        cid_rand +=  sc.lempel_ziv_complexity(rand_seq, "lz77")[2]
    end
    cid_rand = cid_rand / niter / L
    return cid_rand
end

function get_entropy_rate(c, nsites, norm)
    """
    compute entropy rate
    """
    h = (c * log2(c) + 2 * c * log2(nsites / c)) / nsites
    h /= norm
    return h
end

function get_entropy_lz77(lattice, nsites, d, sc)
    """
    """
    rand_seq = rand(d, nsites)
    c_bin, sumlog_bin = sc.lempel_ziv_complexity(rand_seq, "lz77")
    h_bound_bin = get_entropy_rate(c_bin, nsites, 1)
    h_sumlog_bin = sumlog_bin
    c, h_sumlog = sc.lempel_ziv_complexity(lattice, "lz77")
    h_bound = get_entropy_rate(c, nsites, h_bound_bin)
    h_sumlog /= h_sumlog_bin
    return h_bound, h_sumlog
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

function get_exact_entropy(T)
    """
    computes exact entropy of the system
    """
    s = (get_exact_energy(T) - get_exact_free_energy(T)) / T
    return s
end

function plot_data_entropy(
    L,
    T_exact,
    S_exact,
    S_sim,
    T_sim,
    plotspath
    )
    s_plot = plot(
        T_exact,
        S_exact,
        label = "exact",
        c = "maroon",
        minorticks=:false,
        tick_direction = :out,
        framestyle=:box,
        thickness_scaling=1.6,
        grid=:none,
        guidefontsize=12,
    )
    shading = 0.5
    scatter!(s_plot,
        T_sim,
        S_sim,
        xlabel=L"T",
        ylabel=L"S/N",
        # ribbon = StDev,
        # fillalpha = shading,
        # yerror = StDev,
        minorticks=:false,
        tick_direction = :out,
        framestyle=:box,
        thickness_scaling=1.6,
        grid=:none,
        guidefontsize=12,
        legend = :bottomright,
        legendfontsize=6,
        label = "LZ-77",
        c = "lightblue"
    )
    savefig(s_plot, plotspath*"2D_ising_entropy"*string(L)*".png")
    # return s_plot
end

function get_s(T)
    S_exact = zeros(length(T), 1)
    for (idx, temp) in enumerate(T)
        S_exact[idx] = get_exact_entropy(temp)
    end
    return S_exact
end

L = 512
T = 1:0.1:5
ntemp = 41
nconfig = 1
cd("/home/mart5523/ribei040/IsingModelJulia")
# sc = pyimport("sweetsourcod.lempel_ziv")
ising_repo_path = pwd()
# today_date = string(today())
# mkpath("Simulation_Results/"*today_date*"/configs/")
# mkpath("Simulation_Results/"*today_date*"/plots/wolff/")
# configs_path = ising_repo_path*"/Simulation_Results/"*today_date*"/configs/"
# configs_path = "/home/mart5523/ribei040/IsingModelJulia/Simulation_Results/2022-03-21/configs/"
# plots_path = ising_repo_path*"/Simulation_Results/"*today_date*"/plots/metropolis/"
plots_path = "/home/mart5523/ribei040/IsingModelJulia/Simulation_Results/2022-03-21/"
h_data_fname = "entropy_data.txt"
h_path = "/home/mart5523/ribei040/IsingModelJulia/Simulation_Results/"
final_data = get_stats(h_data_fname, h_path, ntemp, nconfig)
S_exact = get_s(T)
plot_data_entropy(
    L,
    T,
    S_exact,
    final_data[:,1],
    1:0.1:5,
    plots_path
)

# using Distributions

# nsites = 128^2
# p = 0:0.01:1.00
# H = zeros(length(p), 1)
# cid_rand = get_cid_rand(nsites, 10, sc)
# for (idx,prob) in ProgressBar(enumerate(p))
#     d = Bernoulli(prob)
#     rand_seq = rand(d, nsites)
#     H[idx] = sc.lempel_ziv_complexity(rand_seq, "lz77")[2] / nsites
# end
#
# plot(p, H)

# L = (250)^2
# niter = 100000
# cids = zeros(niter, 1)
# for i = ProgressBar(1:niter)
#     rand_seq = rand([0,1], L)
#     cid_rand =  sc.lempel_ziv_complexity(rand_seq, "lz77")[2]
#     cids[i] = cid_rand
# end
# avg = mean(cids)
# sigma = std(cids)
# println("-----")
# println("mean = " * string(avg))
# println("stdev = " * string(sigma / avg))
