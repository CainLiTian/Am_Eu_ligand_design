from plot import plot_baseline, plot_reward_comparison, create_fingerprint_visualizations
import pandas as pd

#random_df = pd.read_excel('/home/dc/data_new/ppo_results/baseline/random.xlsx')
#greedy_df = pd.read_excel('/home/dc/data_new/ppo_results/baseline/Greedy.xlsx')
#cem_df = pd.read_excel('/home/dc/data_new/ppo_results/baseline/cem.xlsx')
ppo_df = pd.read_excel('/home/dc/data_new/ppo_results/22/training_log.xlsx', sheet_name='Sheet1')
#data_df = pd.read_excel('/home/dc/data_new/S.xlsx',SheetName='MAIN')


# plot_baseline(random_df, '/home/dc/data_new/ppo_results/baseline/random_plot.png')
# plot_baseline(greedy_df, '/home/dc/data_new/ppo_results/baseline/greedy_plot.png')
# plot_baseline(cem_df, '/home/dc/data_new/ppo_results/baseline/cem_plot.png')
#plot_baseline(ppo_df, '/home/dc/data_new/ppo_results/22/ppo_plot.png')


#plot_reward_comparison(ppo_df, random_df, greedy_df, cem_df, '/home/dc/data_new/ppo_results/22/comparison.png', window_size=3, total_step = 100)

create_fingerprint_visualizations(ppo_df,'/home/dc/data_new/ppo_results/22/')

