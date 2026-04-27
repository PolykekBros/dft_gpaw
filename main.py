from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from gpaw import GPAW, PW, FermiDirac
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read
from ase.visualize import view
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = "out/"
PSEUDODIR = "./pot/"
QEDIR = "/Users/kirill/Desktop/src/q-e-qe-7.5/bin/pw.x"


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


def hypercell_init(s, primitive_atoms):
    hc_atoms = primitive_atoms * (s, s, 1)
    print("Гиперячейка создана!")

    cell_info = hc_atoms.cell.cellpar()
    print(f"a = {cell_info[0]} Å, b = {cell_info[1]} Å, c = {cell_info[2]} Å")
    print(f"alpha = {cell_info[3]}, beta = {cell_info[4]}, gamma = {cell_info[5]}")
    view(hc_atoms)

    return hc_atoms


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


def calculate_band_gap_qe(atoms, u, output_qe):
    if not os.path.exists(output_qe):
        os.makedirs(output_qe)
    pseudopotentials = {
        "Ni": "Ni.pbesol-n-rrkjus_psl.0.1.UPF",
        "O": "O.pbesol-n-rrkjus_psl.1.0.0.UPF",
    }
    profile = EspressoProfile(command=QEDIR, pseudo_dir=PSEUDODIR)
    input_data = {
        "control": {
            "calculation": "scf",
        },
        "system": {
            "ecutwfc": 55,
            "ecutrho": 550,
            "nbnd": 40,
            "tot_magnetization": 0.0,
        },
        "electrons": {
            "conv_thr": 1e-8,
        },
    }
    additional_cards = ["HUBBARD (ortho-atomic)", f"U Ni-3d {u}", f"U Ni1-3d {u}"]
    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        additional_cards=additional_cards,
        kpts=(4, 4, 4),
        directory=output_qe,
    )
    atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except Exception:
        print("Error getting potential energy")

    homo, lumo, band_gap = get_band_gap_qe(f"{output_qe}/espresso.pwo")
    print(f"Верхняя граница валентной зоны (HOMO): {homo:.3f} eV")
    print(f"Нижняя граница зоны проводимости (LUMO): {lumo:.3f} eV")
    print(f"Ширина запрещенной зоны (Band Gap): {band_gap:.3f} eV")
    print("Расчет завершен.")

    return band_gap


def get_band_gap_qe(filename="espresso.pwo"):
    homo = None
    lumo = None

    try:
        with open(filename, "r") as f:
            for line in f:
                if "highest occupied, lowest unoccupied level" in line:
                    parts = line.split()
                    homo = float(parts[-2])
                    lumo = float(parts[-1])

        if homo is not None and lumo is not None:
            return homo, lumo, lumo - homo
        else:
            print(f"Warning: HOMO/LUMO summary line not found in {filename}")
            return None

    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None


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


def plotter(plt_data, title, x_label, y_label, filename):
    data = np.loadtxt(plt_data)
    x = data[:, 0]
    y = data[:, 1]

    plt.plot(x, y, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    plt.savefig(filename)


# Основной блок выполнения
if __name__ == "__main__":
    opt_traj_nio = f"{OUTDIR}opt_nio.traj"
    nio_structure = afm_nio_init()
    print(f"Current symbols: {nio_structure.get_chemical_symbols()}")
    print(f"Current tags: {nio_structure.get_tags()}")
    optimized_nio = stuct_optim(
        nio_structure, f"{OUTDIR}optimization.txt", opt_traj_nio
    )
    struct_analyze(optimized_nio)
    nio_hypercell = hypercell_init(2, optimized_nio)

    qe_gaps = []
    for u in np.arange(3.0, 8.1, 0.5):
        print(f"Calculating band gap for u = {u}")
        qe_gap = calculate_band_gap_qe(nio_hypercell, u, f"{OUTDIR}qe/nio_bands_{u}")
        qe_gaps.append([u, qe_gap])

    print(qe_gaps)
    with open("bands_qe.txt", "w") as f:
        for row in qe_gaps:
            line = " ".join(map(str, row))
            f.write(line + "\n")

    plotter(
        "bands_qe.txt",
        "QE band gap VS Habbard U",
        "Habbard U, eV",
        "Band gap, eV",
        "bands_qe.png",
    )

    gaps = []
    for u in np.arange(3.0, 8.1, 0.5):
        print(f"Calculating band gap for u = {u}")
        gap = calculate_band_gap(nio_hypercell, u, f"{OUTDIR}gpaw/nio_bands_{u}.txt")
        gaps.append([u, gap])

    print(gaps)
    with open("bands_gpaw.txt", "w") as f:
        for row in gaps:
            line = " ".join(map(str, row))
            f.write(line + "\n")

    plotter(
        "bands_gpaw.txt",
        "GPAW band gap VS Habbard U",
        "Habbard U, eV",
        "Band gap, eV",
        "bands_gpaw.png",
    )
