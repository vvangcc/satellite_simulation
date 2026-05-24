import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse

from matplotlib import rcParams

rcParams['font.family'] = ['Times New Roman','SimSun']
rcParams['axes.unicode_minus'] = False

def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(len(sorted_data)) / float(len(sorted_data))
    return sorted_data, cdf

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    file_paths = [entry['path'] for entry in config['files']]
    names = [entry['name'] for entry in config['files']]
    save_plots = config.get('save_plots', False)
    return file_paths, names, save_plots

def read_satellite_times(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        last_line = file.readlines()[-1].strip()
    data_dict = eval(last_line)
    times = list(data_dict.values())
    return times

def plot_cdfs(file_paths, names, colors, line_styles, save_plots=False):
    plt.figure(figsize=(4, 3), dpi=400)

    for i, (file_path, name) in enumerate(zip(file_paths, names)):
        times = read_satellite_times(file_path)
        sorted_data, cdf = compute_cdf(times)
        sorted_data = np.append(sorted_data, 600)
        cdf = np.append(cdf, 1)
        plt.plot(sorted_data, cdf, label=name, linestyle=line_styles[i], color=colors[i])

    plt.xlabel('卫星计算时间 (s)')
    plt.ylabel('累计分布函数')
    plt.xlim(left=0, right=600)
    plt.ylim(bottom=0, top=1.4)
    plt.legend()
    plt.yticks([i / 5 for i in range(6)])
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_plots:
        os.makedirs("./images", exist_ok=True)
        plt.savefig(f"./images/satellite_computing_time_CDF_CN.pdf", format='pdf')
        plt.savefig(f"./images/satellite_computing_time_CDF_CN.png", format='png')
    plt.show()

line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
colors = ['#bcbd22', '#7f7f7f', '#17becf', '#1f77b4', '#e377c2','#9467bd']

parser = argparse.ArgumentParser(description='Plot CDFs from satellite times.')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
args = parser.parse_args()

file_paths, names, save_plots = load_config(args.config)

plot_cdfs(file_paths, names, colors[:len(file_paths)], line_styles[:len(file_paths)], save_plots)
