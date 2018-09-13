#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a simulation and save outputs

Authors:
    - Andrea Maiani <andrea.maiani@mail.polimi.it>
    - Ciro Pentangelo <ciro.pentangelo@mail.polimi.it>
"""

import os
import time

import simulations_collection as sim
from waveplotter import WavePlotter

if __name__ == "__main__":
    #####CALL BELOW THE SIMULATION INIT FUNCTION#####
    ss, x0, p0 = sim.dancing()
    #################################################

    wp = WavePlotter(ss, x0, p0)

    print("+----------------------------+")
    print("|     QuantumWave            |")
    print("+----------------------------+")
    print("")
    print("")

    print("SIMULATION NAME: " + ss.simname)
    print("[*] Running numerical simulation...   ", end='')
    start = time.time()
    ss.run()
    end = time.time()
    print("completed in %.2fs" % (end - start))

    print("[*] Building output...   ")
    os.makedirs("./output/" + ss.simname, exist_ok=True)

    print("    [Norm]", end='')
    start = time.time()
    fig = wp.plot_norm()
    fig.savefig("./output/" + ss.simname + "/" + ss.simname + "-norm.pdf")
    end = time.time()
    print("     completed in %.2fs" % (end - start))

    print("    [Optics]", end='')
    start = time.time()
    fig = wp.plot_optics()
    fig.savefig("./output/" + ss.simname + "/" + ss.simname + "-optics.png")
    end = time.time()
    print("     completed in %.2fs" % (end - start))

    print("    [Animation]", end='')
    start = time.time()
    wp.createmp4("./output/" + ss.simname + "/" + ss.simname + "-animation.mp4", center_line=False, max_line=False)
    end = time.time()
    print("     completed in %.2fs" % (end - start))

    print("[*] Execution completed")
