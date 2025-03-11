import click
from runs import RunFactory


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--run', '-r', default='adult_fedavg', help='Run to execute')
@click.option('--project_name', '-p', default='AdultFedAvg', help='Project name')
@click.option('--start_index', '-s', default=1, help='Start index')
@click.option('--metric_name', '-m', default='demographic_parity', help='Metric name')
@click.option('--id', '-i', default='test', help='Run id')
@click.option('--group_name', '-g', default='Gender', help='Sensitive Attribute')
@click.option('--use_hale', '-h', is_flag=True, help='Use HALE')
@click.option('--onlyperf', '-o', is_flag=True, help='Monitor only performance')
@click.option('--threshold', '-t', default=0.2, help='Fairness threshold')
@click.option('-metrics_list', '-ml', multiple=True, help='List of metrics')
@click.option('-groups_list', '-gl', multiple=True, help='List of groups')
@click.option('-threshold_list', '-tl', type=float, multiple=True, help='List of threshold')
@click.option('--num_subproblems', '-ns', default=10, help='Number of subproblems')
@click.option('--num_global_iterations', '-ng', default=30, help='Number of global iterations')
@click.option('--num_local_iterations', '-nl', default=30, help='Number of local iterations')
@click.option('--performance_constraint', '-pc',default=1.0, help='Performance constraint')
@click.option('--delta', '-d', default=0.1, help='Delta')
@click.option('--max_constraints', '-mc', default=10000, help='Max constraints')
@click.option('--global_patience', '-gp', default=5, help='Global patience')
def main(run, project_name, start_index, metric_name, id,
         group_name, use_hale, onlyperf, threshold,
         metrics_list, groups_list, threshold_list,
         num_subproblems, num_global_iterations, num_local_iterations,performance_constraint,delta,
         max_constraints,global_patience):

    run = RunFactory.create_run(run,
                                project_name=project_name,
                                start_index=start_index,
                                metric_name=metric_name,
                                id=id,
                                group_name=group_name,
                                use_hale=use_hale,
                                onlyperf=onlyperf,
                                threshold=threshold,
                                metrics_list=metrics_list,
                                groups_list=groups_list,
                                threshold_list=threshold_list,
                                num_subproblems=num_subproblems,
                                num_global_iterations=num_global_iterations,
                                num_local_iterations=num_local_iterations,
                                performance_constraint=performance_constraint,
                                delta=delta,
                                max_constraints_in_subproblem=max_constraints,
                                global_patience=global_patience,
                                )
    run()


if __name__ == '__main__':
    # mp.set_start_method("spawn", force=True)
    main()
