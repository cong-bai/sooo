import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("cong-b/icml/25nq6eo8")

# save the metrics for the run to a csv file
metrics_dataframe = run.scan_history(keys=["iter_time"])
time = [row["iter_time"] for row in metrics_dataframe][2:]
time1 = [t for t in time if 0.83 < t < 0.86]
time2 = [t for t in time if t > 1.5]
print(sum(time1) / len(time1), sum(time2) / len(time2))
metrics_dataframe.to_csv("metrics.csv")
