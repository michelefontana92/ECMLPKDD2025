import click
from runs import RunFactory


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--run', '-r', default='compas_fairlab', help='Run to execute')
@click.option('--project_name', '-p', default='CompasFairLab', help='Project name')
@click.option('--start_index', '-s', default=1, help='Start index')
@click.option('--id', '-i', default='test', help='Run id')
@click.option('-metrics_list', '-ml', multiple=True, help='List of metrics')
@click.option('-groups_list', '-gl', multiple=True, help='List of groups')
@click.option('-threshold_list', '-tl', type=float, multiple=True, help='List of threshold')
@click.option('--num_subproblems', '-ns', default=10, help='Number of subproblems')
@click.option('--num_global_iterations', '-ng', default=30, help='Number of global iterations')
@click.option('--num_local_iterations', '-nl', default=30, help='Number of local iterations')
@click.option('--performance_budget', '-pb',default=1.0, help='Performance constraint')
@click.option('--delta', '-d', default=0.02, help='Delta')
@click.option('--delta_step', '-ds', default=0.01, help='Delta')
@click.option('--delta_tol', '-dt', default=0.05, help='Delta')
@click.option('--max_constraints', '-mc', default=10000, help='Max constraints')
@click.option('--global_patience', '-gp', default=5, help='Global patience')
def main(run, project_name, start_index, id,
         metrics_list, groups_list, threshold_list,
         num_subproblems, num_global_iterations, 
         num_local_iterations,performance_budget,
         delta,delta_step,delta_tol,
         max_constraints,global_patience):

    run = RunFactory.create_run(run,
                                project_name=project_name,
                                start_index=start_index,
                                id=id,
                                metrics_list=metrics_list,
                                groups_list=groups_list,
                                threshold_list=threshold_list,
                                num_subproblems=num_subproblems,
                                num_global_iterations=num_global_iterations,
                                num_local_iterations=num_local_iterations,
                                performance_constraint=performance_budget,
                                delta=delta,
                                delta_step=delta_step,
                                delta_tol=delta_tol,
                                max_constraints_in_subproblem=max_constraints,
                                global_patience=global_patience,
                                
                                )
    run()


if __name__ == '__main__':
    # mp.set_start_method("spawn", force=True)
    main()
