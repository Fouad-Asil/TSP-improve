import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import move_to, clip_grad_norms, get_inner_model
from utils.logger import log_to_screen, log_to_tb_train, log_to_tb_val

def rollout(problem, model, x_input, batch, solution, value, opts, T, do_sample = False, record = False):
    
    solutions = solution.clone()
    best_so_far = solution.clone()
    cost = value
    
    exchange = None
    best_val = cost.clone()
    improvement = []
    reward = []
    solution_history = [best_so_far]
    
    for t in tqdm(range(T), disable = opts.no_progress_bar, desc = 'rollout', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        
        exchange, _ = model( x_input, 
                             solutions, 
                             exchange, 
                             do_sample = do_sample)
        
        # new solution
        solutions = problem.step(solutions, exchange)        
        solutions = move_to(solutions, opts.device)
        
        obj = problem.get_costs(batch, solutions)
        
        #calc improve
        improvement.append(cost - obj)
        cost = obj
        
        #calc reward
        new_best = torch.cat((best_val[None,:], obj[None,:]),0).min(0)[0]
        r = best_val - new_best
        reward.append(r)        
        
        #update best solution
        best_val = new_best
        best_so_far[(r > 0)] = solutions[(r > 0)]
        
        #record solutions
        if record: solution_history.append(best_so_far.clone())
        
    return best_val.view(-1,1), torch.stack(improvement,1), torch.stack(reward,1), None if not record else torch.stack(solution_history,1)

def validate(problem, model, val_dataset, tb_logger, opts, _id = None):
    # Validate mode
    print('\nValidating...', flush=True)
    model.eval()
    
    init_value = []
    best_value = []
    improvement = []
    reward = []
    time_used = []
    
    for batch in tqdm(DataLoader(val_dataset, batch_size = opts.eval_batch_size), 
                        disable = opts.no_progress_bar or opts.val_size == opts.eval_batch_size, 
                        desc = 'validate', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        
        #initial solutions
        initial_solution = move_to(
                problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
        
        if problem.NAME == 'tsp':
            x = batch
            
        else:
            assert False, "Unsupported problem: {}".format(problem.NAME)
        
        x_input = move_to(x, opts.device) # batch_size, graph_size, 2
        batch = move_to(batch, opts.device) # batch_size, graph_size, 2
        
        initial_value = problem.get_costs(batch, initial_solution)
        init_value.append(initial_value)
        
        # run the model
        s_time = time.time()
        bv, improve, r, _  = rollout(problem, 
                                     model, 
                                     x_input,
                                     batch, 
                                     initial_solution,
                                     initial_value,
                                     opts,
                                     T=opts.T_max,
                                     do_sample = True)
        
       
        duration = time.time() - s_time
        time_used.append(duration)
        best_value.append(bv.clone())
        improvement.append(improve.clone())
        reward.append(r.clone())

    best_value = torch.cat(best_value,0)
    improvement = torch.cat(improvement,0)
    reward = torch.cat(reward,0)
    init_value = torch.cat(init_value,0).view(-1,1)
    time_used = torch.tensor(time_used)
    
    # log to screen
    log_to_screen(time_used, 
                  init_value, 
                  best_value, 
                  reward, 
                  improvement, 
                  batch_size = opts.eval_batch_size, 
                  dataset_size = len(val_dataset), 
                  T = opts.T_max)
    
    # log to tb
    if(not opts.no_tb):
        log_to_tb_val(tb_logger,
                      time_used, 
                      init_value, 
                      best_value, 
                      reward, 
                      improvement, 
                      batch_size = opts.eval_batch_size, 
                      dataset_size = len(val_dataset), 
                      T = opts.T_max,
                      epoch = _id)
    
    # save to file
    if _id is not None:
        save_path = os.path.join(opts.save_dir, f'validate-{_id}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        torch.save(
        {
            'init_value': init_value,
            'best_value': best_value,
            'improvement': improvement,
            'reward': reward,
            'time_used': time_used,
        },
        save_path)
        
    

def train_epoch(problem, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts):
    
    # lr_scheduler
    lr_scheduler.step(epoch)
    
    print('\n\n')
    print("|",format(f" Training epoch {epoch} ","*^60"),"|")
    print("Training with lr={:.3e} for run {}".format(optimizer.param_groups[0]['lr'], opts.run_name), flush=True)
    step = epoch * (opts.epoch_size // opts.batch_size)    

    # Generate new training data for each epoch
    training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

    # Put model in train mode!
    model.train()

    # start training  
    pbar = tqdm(total = (opts.epoch_size // opts.batch_size) * (opts.T_train // opts.n_step) ,
                disable = opts.no_progress_bar, desc = f'training',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    for batch_id, batch in enumerate(training_dataloader):
        
        train_batch(
                problem,
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts,
                pbar
            )

        step += 1
    pbar.close()

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        save_path = os.path.join(opts.save_dir, f'epoch-{epoch}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            save_path)
    
    validate(problem, model, val_dataset, tb_logger, opts, _id = epoch)
          
def train_batch(
        problem,
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
        pbar
):
    
    solution = move_to(
        problem.get_initial_solutions(opts.init_val_met, batch), opts.device)


    if problem.NAME == 'tsp':
        x = batch
        
    else:
        assert False, "Unsupported problem: {}".format(problem.NAME)
    
    x_input = move_to(x, opts.device) # batch_size, graph_size, 2
    batch = move_to(batch, opts.device) # batch_size, graph_size, 2
    exchange = None	
    
    # Get optimal tours if enabled
    optimal_tours = None
    if opts.use_optimal_tours and problem.optimal_tours is not None:
        start_idx = batch_id * opts.batch_size
        # Ensure we don't slice beyond the available optimal tours
        end_idx = min(start_idx + opts.batch_size, len(problem.optimal_tours))
        # Only proceed if there are tours for this slice
        if start_idx < end_idx:
            optimal_tours_slice = problem.optimal_tours[start_idx:end_idx]
            # Ensure the slice matches the actual batch size being processed
            current_batch_size = batch.size(0) # Get the real size of this batch
            if len(optimal_tours_slice) == current_batch_size:
                 optimal_tours = move_to(torch.tensor(optimal_tours_slice), opts.device)
            else:
                 # Handle mismatch - perhaps skip overlap calculation for this batch or log a warning
                 print(f"Warning: Batch size ({current_batch_size}) and optimal tours slice size ({len(optimal_tours_slice)}) mismatch for batch_id {batch_id}. Skipping overlap calculation.")
                 optimal_tours = None # Prevent using mismatched tours

    #update best_so_far
    best_so_far = problem.get_costs(batch, solution)
    initial_cost = best_so_far.clone()
    
    # Calculate initial edge overlap if using optimal tours
    prev_overlap = None
    if optimal_tours is not None:
        # Ensure the solution batch size matches optimal_tours batch size before calculating
        if solution.size(0) == optimal_tours.size(0):
             prev_overlap = problem.calculate_edge_overlap(solution, optimal_tours)
        else:
             print(f"Warning: Mismatch between solution batch size ({solution.size(0)}) and optimal_tours batch size ({optimal_tours.size(0)}). Skipping overlap calculation.")
             optimal_tours = None # Ensure downstream code doesn't use mismatched tours

    # params
    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    t = 0
    
    while t < T:
    
        baseline_val = []
        baseline_val_detached = []
        log_likelihood = []
        reward = []
        overlap_rewards = []
        penalty_rewards = []
        
        t_s = t
        
        total_cost = 0
        exchange_history = []              
        
        while t - t_s < n_step and not (t == T):
            
            
            # get estimated value from baseline
            bl_val_detached, bl_val = baseline.eval(x_input, solution)
            
            baseline_val_detached.append(bl_val_detached)
            baseline_val.append(bl_val)
            
            # get model output
            exchange, log_lh = model( x_input, 
                                      solution,
                                      exchange, 
                                      do_sample = True)
            
            exchange_history.append(exchange)
            log_likelihood.append(log_lh)
            
            # Calculate penalties for breaking optimal edges if using optimal tours
            penalty = torch.zeros_like(best_so_far)
            if optimal_tours is not None and opts.break_penalty_weight > 0:
                 # Add similar size check here
                 if solution.size(0) == optimal_tours.size(0):
                     broken_edges = problem.calculate_broken_optimal_edges(solution, exchange, optimal_tours)
                     penalty = opts.break_penalty_weight * broken_edges
                 else:
                     # Handle mismatch if necessary, maybe penalty remains zero
                     print(f"Warning: Mismatch during penalty calculation. Sol size: {solution.size(0)}, Opt size: {optimal_tours.size(0)}")

            # state transient
            solution = problem.step(solution, exchange)
            solution = move_to(solution, opts.device)
        
            # calc reward
            cost = problem.get_costs(batch, solution)
            total_cost = total_cost + cost
            best_for_now = torch.cat((best_so_far[None,:], cost[None,:]),0).min(0)[0]
            
            # Calculate edge overlap reward if using optimal tours
            overlap_reward = torch.zeros_like(best_so_far)
            if optimal_tours is not None and opts.overlap_reward_weight > 0:
                 # Add similar size check here
                 if solution.size(0) == optimal_tours.size(0):
                     curr_overlap = problem.calculate_edge_overlap(solution, optimal_tours)
                     # Check if prev_overlap was calculated for this batch size
                     if prev_overlap is not None and prev_overlap.size(0) == curr_overlap.size(0):
                         overlap_reward = opts.overlap_reward_weight * (curr_overlap - prev_overlap)
                         prev_overlap = curr_overlap # Update prev_overlap only if calculated
                     else:
                         # Handle case where prev_overlap wasn't set correctly for this batch
                         print(f"Warning: Mismatch or missing prev_overlap during overlap reward calculation.")
                         # Decide how to handle this: recalculate prev_overlap or skip reward?
                         # Option 1: Recalculate prev_overlap (might be slightly off if solution changed significantly)
                         # prev_overlap = problem.calculate_edge_overlap(solution, optimal_tours)
                         # Option 2: Skip overlap reward calculation for this step
                         overlap_reward = torch.zeros_like(best_so_far) # Keep reward zero

                 else:
                     print(f"Warning: Mismatch during overlap reward calculation. Sol size: {solution.size(0)}, Opt size: {optimal_tours.size(0)}")

            # Calculate standard improvement reward
            improvement_reward = best_so_far - best_for_now
            
            # Combine rewards: improvement + overlap reward - edge breaking penalty
            combined_reward = improvement_reward + overlap_reward - penalty
            reward.append(combined_reward)
            
            # Update best_so_far for next iteration
            best_so_far = best_for_now
            
            # next            
            t = t + 1
        
        # Get discounted R
        Reward = []
        
        total_cost = total_cost / (t-t_s)
        
        reward_reversed = reward[::-1]
        next_return, _  = baseline.eval(x_input, solution)

        for r in range(len(reward_reversed)):     
            R = next_return * gamma + reward_reversed[r]            
            Reward.append(R)
            next_return = R       
        
        Reward = torch.stack(Reward[::-1], 0)
        baseline_val = torch.stack(baseline_val,0)
        baseline_val_detached = torch.stack(baseline_val_detached,0)
        log_likelihood = torch.stack(log_likelihood,0)

        # calculate loss
        criteria = torch.nn.MSELoss() 
        baseline_loss = criteria(Reward, baseline_val)
        reinforce_loss = - ((Reward - baseline_val_detached)*log_likelihood).mean()
        loss =  baseline_loss + reinforce_loss
        
        # update gradient step
        optimizer.zero_grad()
        loss.backward()
        
        #Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        
        optimizer.step()
    
        # Logging to tensorboard
        if(not opts.no_tb):
            current_step = int(step * T / n_step + t // n_step)
            if current_step % int(opts.log_step) == 0:
                # Prepare overlap rewards and penalties for logging if they exist
                overlap_rewards_data = None
                if overlap_rewards:
                    overlap_rewards_data = torch.stack(overlap_rewards, 0)
                    
                penalty_rewards_data = None
                if penalty_rewards:
                    penalty_rewards_data = torch.stack(penalty_rewards, 0)
                
                log_to_tb_train(tb_logger, optimizer, model, baseline, total_cost, grad_norms, reward, 
                   exchange_history, reinforce_loss, baseline_loss, log_likelihood, initial_cost, current_step,
                   overlap_rewards=overlap_rewards_data, penalty_rewards=penalty_rewards_data)
        
        pbar.update(1)