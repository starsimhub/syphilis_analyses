####### PLOT RESULTS
import matplotlib.pyplot as plt
import numpy as np
import stisim as ss
from stisim import interventions as ssi
import sciris as sc


def plot_cum_infections(sim, ax, disease):
    if sim.results[disease]["cum_infections"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["cum_infections"].low, sim.results[disease]["cum_infections"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["cum_infections"][:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " infections")

def plot_cum_diagnoses(sim, ax, disease):
    if sim.results[disease]["cum_diagnoses"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["cum_diagnoses"].low, sim.results[disease]["cum_diagnoses"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["cum_diagnoses"][:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " diagnoses")

def plot_cum_deaths(sim, ax, disease):
    if sim.results[disease]["cum_deaths"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["cum_deaths"].low, sim.results[disease]["cum_deaths"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["cum_deaths"][:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " deaths")

def plot_new_infections(sim, ax, disease):
    if sim.results[disease]["new_infections"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["new_infections"].low, sim.results[disease]["new_infections"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["new_infections"][:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " infections")


def plot_new_diagnoses(sim, ax, disease):
    if sim.results[disease]["new_diagnoses"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["new_diagnoses"].low, sim.results[disease]["new_diagnoses"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["new_diagnoses"][:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " diagnoses")


def plot_new_deaths(sim, ax, disease):
    if sim.results[disease]["new_deaths"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["new_deaths"].low, sim.results[disease]["new_deaths"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["new_deaths"][:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " deaths")
def plot_prevalence(sim, ax, disease='hiv'):
    if sim.results[disease]["prevalence"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["prevalence"].low, sim.results[disease]["prevalence"].high, **fill_args)

    ax.plot(sim.tivec, sim.results[disease]["prevalence"][:], color="b", alpha=1)

    ax.set_title("Prevalence of HIV in the population")

def diagnostic_plots(sim, disease):

    # MAIN FIGURE
    fig, ax = plt.subplots(2, 2)

    fig.set_size_inches(10, 10)
    fig.tight_layout(pad=5.0)
    plot_cum_infections(sim, ax[0, 0], disease)
    plot_cum_diagnoses(sim, ax[0, 1], disease)
    #plot_cum_tests(sim, ax[1, 0])
    plot_cum_deaths(sim, ax[1, 0], disease)
    #plot_cum_sb(sim, ax[1, 0])
    plot_prevalence(sim, ax[1, 1], disease)

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Key outputs")
    fig.savefig(disease + '_output_check', transparent=True)

