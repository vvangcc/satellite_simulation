import os
import re
import matplotlib.pyplot as plt

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

    return {
        'Average Ending Reward': avg_ending_rewards
    }


def plot_data(folder_paths, names, colors, linestyles):
    attribute = 'Average Ending Reward'

    # 'Proportion in Computation', 'Packet Loss Rate', 'Average Delay', 'Average Hop Count', 'Average Waiting Time'


    plt.figure(figsize=(4, 3), dpi= 400)
    min_totals,max_totals=None,None
    for folder_path, name, color, linestyle in zip(folder_paths, names, colors, linestyles):
        values = []
        if os.path.isfile(folder_path) and folder_path.endswith('.txt'):
            parsed_data = parse_statics(folder_path)
            values.extend(parsed_data[attribute])
        steps = [10 * (i + 1) for i in range(len(values))]

        plt.plot(steps, values, color = color, label=name, linestyle=linestyle)
        min_totals, max_totals = min(steps), max(steps)
    plt.xlim(min_totals, max_totals)

    plt.xlabel('Simulation time (s)')
    plt.ylabel(attribute)
    # plt.title(f'{attribute} vs Average Total')
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"../images/training.pdf", format='pdf')
    plt.savefig(f"../images/training.png", format='png')
    plt.show()

colors = ['#7f7f7f', '#17becf', '#bcbd22', '#1f77b4', '#e377c2','#9467bd']
linestyles = [':' , '--', '-', '-.', (0, (3, 5, 1, 5))]
folder_paths = ["../training_process_data/TCOM_version/train/train_NewDDQN_dueling_shuffle.txt",'../training_process_data/TCOM_version/train/train_PurePPO_shuffle.txt','../training_process_data/TCOM_version/train/train_PureDDQN_dueling.txt','../training_process_data/TCOM_version/train/train_PureDDQN.txt']
names = ['DDQN','PPO','D3QN','G-D3QN']
plot_data(folder_paths, names, colors[:len(folder_paths)], linestyles[:len(folder_paths)])