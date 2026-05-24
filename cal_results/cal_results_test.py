import re

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
        'Average Total': calculate_average(total_values),
        'Average Packet Loss Rate': calculate_average(packet_loss_rates),
        'Average Delay for Successful Transmissions': calculate_average(avg_delays),
        'Average Hop Count for Successful Transmissions': calculate_average(avg_hop_counts),
        'Proportion of Satellites in Computation': calculate_average(proportions_in_computation),
        'Average Waiting Time for Computing': calculate_average(avg_waiting_times),
        'Average Ending Reward': calculate_average(avg_ending_rewards)
    }

file_path = ('../training_process_data/VTC_version/dif_load/DDQN_ds_m3/test_PureDDQN_dueling_shuffle_25_m3.txt')
results = parse_statics(file_path)

for key, value in results.items():
    print(f"{key}: {value:.2f}")
