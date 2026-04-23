from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from gpaw import GPAW, PW, FermiDirac
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "out/"


def afm_nio_init():
    """
    Создает суперячейку NiO с антиферромагнитным упорядочением.
    Используется примитивная ячейка, расширенная для размещения двух магнитных подрешеток.
    """
    nio = bulk("NiO", "rocksalt", a=4.17)
    nio = nio * (1, 1, 2)
    magmoms = [2.0, 0.0, -2.0, 0.0]
    nio.set_initial_magnetic_moments(magmoms)

    print("Структура NiO AFM успешно инициализирована.")

    cell_info = nio.cell.cellpar()
    print(f"a = {cell_info[0]} Å, b = {cell_info[1]} Å, c = {cell_info[2]} Å")
    print(f"alpha = {cell_info[3]}, beta = {cell_info[4]}, gamma = {cell_info[5]}")

    return nio


def stuct_optim(atoms, calc_txt, opt_traj):
    """
    Выполняет релаксацию ионов и оптимизацию параметров решетки.
    """
    calc = GPAW(
        mode=PW(500),  # Энергия отсечки плоских волн
        xc="PBE",  # Функционал PBE
        setups={"Ni": ":d,4.6"},  # DFT+U коррекция
        kpts={"size": (4, 4, 4), "gamma": True},
        occupations=FermiDirac(0.01),
        txt=calc_txt,
    )

    atoms.calc = calc

    # Фильтр для оптимизации формы и объема ячейки одновременно с ионами
    uf = FrechetCellFilter(atoms)
    opt = BFGS(uf, trajectory=opt_traj, logfile="opt_nio.log")
    opt.run(fmax=0.05)

    atoms_opt = read(opt_traj, index="-1")

    print("Оптимизация структуры завершена.")
    return atoms_opt


def calculate_band_gap(atoms, u, output):
    """
    Выполняет статический расчет электронной структуры и извлекает Band Gap.
    """
    calc = GPAW(
        mode=PW(600),
        xc="PBE",
        setups={"Ni": f":d,{u}"},  # DFT+U коррекция
        kpts={"size": (8, 8, 8), "gamma": True},
        occupations=FermiDirac(0.01, fixmagmom=True),
        txt=output,
    )

    atoms.calc = calc
    atoms.get_potential_energy()

    homo, lumo = calc.get_homo_lumo()
    band_gap = lumo - homo
    print("Расчет завершен.")
    print(f"Верхняя граница валентной зоны (HOMO): {homo:.3f} eV")
    print(f"Нижняя граница зоны проводимости (LUMO): {lumo:.3f} eV")
    print(f"Ширина запрещенной зоны (Band Gap): {band_gap:.3f} eV")

    return band_gap


def struct_analyze(atoms):
    """
    Анализирует результаты оптимизации структуры и возвращает новую геометрию
    """
    cell_info = atoms.cell.cellpar()
    forces = atoms.get_forces()
    max_force = (forces**2).sum(axis=1).max() ** 0.5
    # configs = read(output, index=":")
    # energies = [a.get_potential_energy() for a in configs]

    print(f"Max force: {max_force:.4f} eV/Å")
    print(f"a = {cell_info[0]} Å, b = {cell_info[1]} Å, c = {cell_info[2]} Å")
    print(f"alpha = {cell_info[3]}, beta = {cell_info[4]}, gamma = {cell_info[5]}")


def plotter(plt_data):
    data = np.loadtxt(plt_data)
    x = data[:, 0]
    y = data[:, 1]

    plt.plot(x, y)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.title("Imported data plot")

    plt.show()


# Основной блок выполнения
if __name__ == "__main__":
    opt_traj_nio = f"{OUTDIR}opt_nio.traj"
    nio_structure = afm_nio_init()
    optimized_nio = stuct_optim(
        nio_structure, f"{OUTDIR}optimization.txt", opt_traj_nio
    )
    struct_analyze(optimized_nio)

    gaps = []
    for u in np.arange(3.0, 8.1, 0.5):
        print(f"Calculating band gap for u = {u}")
        gap = calculate_band_gap(optimized_nio, u, f"{OUTDIR}nio_bands_{u}.txt")
        gaps.append([u, gap])

    print(gaps)
    with open("bands_gpaw.txt", "w") as f:
        for row in gaps:
            line = " ".join(map(str, row))
            f.write(line + "\n")

    plotter("bands_gpaw.txt")
