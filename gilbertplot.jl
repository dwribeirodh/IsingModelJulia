
using Plots
using GilbertCurves
using LaTeXStrings
using ProgressBars

ls = gilbertindices((2^5,2^5))

# p = plot(
#     [c[1] for c in ls],
#     [c[2] for c in ls],
#     line_z=1:length(ls),
#     minorticks=:false,
#     tick_direction = :out,
#     framestyle=:box,
#     thickness_scaling=1.6,
#     grid=:none,
#     guidefontsize=12,
#     legend = false,
#     xlabel = L"x",
#     ylabel = L"y",
#     legendfontsize=6,
#     # label = "Hilbert Curve",
#     # aspect_ratio = :equal
#     # c = "lightblue"
#     )
# savefig(p,"Hilbert_Scan.png")

freq = 5
a = Animation()
p = plot()
x = []
y = []
for (idx,point) in ProgressBar(enumerate(Tuple.(ls)))
    push!(x, point[1])
    push!(y, point[2])
    if idx % freq == 0
        plot!(x, y, legend = false, line_z=1:length(x))
        frame(a, p)
    end
end
gif(a, "Hilbert.gif", fps = 10)
