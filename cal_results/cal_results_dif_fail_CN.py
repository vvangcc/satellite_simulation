import os
import re
import matplotlib.pyplot as plt
import yaml
import argparse

from matplotlib import rcParams

rcParams['font.family'] = ['Times New Roman','SimSun']
rcParams['axes.unicode_minus'] = False

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    folder_paths = [entry['path'] for entry in config['folders']]
    names = [entry['name'] for entry in config['folders']]
    save_plots = config.get('save_plots', False)
    least_num = config.get('least_num', False)
    return folder_paths, names, save_plots,least_num

def parse_statics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    packet_loss_rates = []
    avg_delays = []
    avg_hop_counts = []
    proportions_in_computation = []
    avg_waiting_times = []
    avg_ending_rewards = []
    total_values = []

    steps = content.split("====== step")
    for step in steps:
        if "current_statics" not in step:
            continue

        total_value = re.search(r"'Total': (\d+)", step)
        packet_loss_rate = re.search(r'Packet loss rate: ([\d\.]+)%', step)
        avg_delay = re.search(r'Average delay for successful transmissions: ([\d\.]+) seconds', step)
        avg_hop_count = re.search(r'Average hop count for successful transmissions: ([\d\.]+) hops', step)
        proportion_in_computation = re.search(r'Proportion of satellites in computation: ([\d\.]+)%', step)
        avg_waiting_time = re.search(r'Average waiting time for computing: ([\d\.]+) seconds', step)
        avg_ending_reward = re.search(r'Average ending reward: ([\d\.]+)', step)

        if total_value:
            total_values.append(int(total_value.group(1)))
        if packet_loss_rate:
            packet_loss_rates.append(float(packet_loss_rate.group(1)))
        if avg_delay:
            avg_delays.append(float(avg_delay.group(1)))
        if avg_hop_count:
            avg_hop_counts.append(float(avg_hop_count.group(1)))
        if proportion_in_computation:
            proportions_in_computation.append(float(proportion_in_computation.group(1)))
        if avg_waiting_time:
            avg_waiting_times.append(float(avg_waiting_time.group(1)))
        if avg_ending_reward:
            avg_ending_rewards.append(float(avg_ending_reward.group(1)))

    def calculate_average(values):
        return sum(values) / len(values) if values else 0

    return {
        'Average Total': calculate_average(total_values) / 10,
        'Packet Loss Rate (%)': calculate_average(packet_loss_rates),
        'Average Delay (s)': calculate_average(avg_delays),
        'Average Hop Count': calculate_average(avg_hop_counts),
        'Proportion in Computation': calculate_average(proportions_in_computation),
        'Average Waiting Time (s)': calculate_average(avg_waiting_times),
        'Average Ending Reward': calculate_average(avg_ending_rewards)
    }

def plot_data(folder_paths, names, colors, markers,save_plots=False, least_num=False):
    attributes = ['Packet Loss Rate (%)', 'Average Delay (s)', 'Average Hop Count', 'Average Waiting Time (s)']
    trans_dict={'Packet Loss Rate (%)': '平均丢包率 (%)',
                'Average Delay (s)': '平均时延 (s)',
                'Average Hop Count': '平均传输跳数',
                'Average Waiting Time (s)': '平均排队时延 (s)',
    }
    # 'Proportion in Computation', 'Average Ending Reward'

    for attribute in attributes:
        plt.figure(figsize=(4, 3), dpi= 400)
        min_totals,max_totals=None,None
        min_length = float('inf')

        data_dict = {}
        for folder_path, name in zip(folder_paths, names):
            avg_failure = []
            values = []
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_path.endswith('.txt'):
                    parsed_data = parse_statics(file_path)
                    avg_failure.append(float(file_name.split('.')[0].split('f')[-1]) * 100 / (12 * 20 * 2))
                    values.append(parsed_data[attribute])

            data_dict[(folder_path, name)] = (avg_failure, values)
            min_length = min(min_length, len(avg_failure), len(values))

        for folder_path, name, color, marker in zip(folder_paths, names, colors, markers):
            avg_failure, values = data_dict[(folder_path, name)]
            if least_num:
                avg_failure = avg_failure[:min_length]
                values = values[:min_length]

            plt.plot(avg_failure, values, label=name, color=color, marker=marker)
            min_totals, max_totals = min(avg_failure), max(avg_failure)

        plt.xlim(min_totals, max_totals)

        plt.xlabel('平均链路失效率 (%)')
        plt.ylabel(trans_dict[attribute])
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        if save_plots:
            os.makedirs("./images", exist_ok=True)
            plt.savefig(f"./images/{attribute}_dif_fail_CN.pdf", format='pdf')
            plt.savefig(f"./images/{attribute}_dif_fail_CN.png", format='png')
        plt.show()

colors = ['#bcbd22', '#7f7f7f', '#17becf', '#1f77b4', '#e377c2','#9467bd']
markers = ['s', '^', 'D', '*', 'X', 'o']

parser = argparse.ArgumentParser(description='Plot data with configuration from a YAML file.')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
args = parser.parse_args()

folder_paths, names, save_plots, least_num = load_config(args.config)

plot_data(folder_paths, names, colors[:len(folder_paths)], markers[:len(folder_paths)],save_plots,least_num)