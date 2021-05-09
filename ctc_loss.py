import torch
from math import inf

class CTCLoss(torch.nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()

    def forward(self, prediction, target, pred_lens, target_lens):
        """
        Forward alpha probabilities calculated using dynamic programming.

        prediction: Tensor of probabilities at each step given input at each step. T * B * C
        target: Tensor indicating the sequence to map input to. B * S
        pred_lens: Tensor indicating length of input. There can be multiple ones in a batch, padded.
        target_lens: Tensor indicating length of output, possibly batched and padded.
        """
        # Create an alpha for each state and populate with prob 0, -inf in log domain
        # The + 1 extra is for the states at the final time step, which don't emit anything
        # b = 0
        # num_states = 2 * target_lens[b] - 1
        # num_steps = pred_lens[b]
        # # T * B * C
        # alphas = torch.full((num_steps, num_states), fill_value=-inf)

        # target-=torch.tensor(1)
        # w_idx = target.tolist()[0]

        # alphas[:,range(1,num_states-1,2)] = torch.transpose(prediction[:num_steps,b,27].repeat(target_lens[b]-1,1),0,1)
        # alphas[:,range(0,num_states,2)] = prediction[:num_steps,b,w_idx]

        num_states = 2 * max(target_lens) - 1
        num_steps = max(pred_lens)
        batch_size = prediction.size()[1]
        # T * B * C
        alphas = torch.full((num_steps, batch_size, num_states), fill_value=-inf)

        target[target==0]+=torch.tensor(1)
        target-=torch.tensor(1)
        w_idx = target.tolist()

        alphas[:,:,range(1,num_states-1,2)] = torch.transpose(prediction[:num_steps,:,27].repeat(target_lens[b]-1,1,1),0,2)
        alphas[:,:,range(0,num_states,2)] = prediction[:num_steps,:,w_idx]

        # For each character
        for tid in range(1,num_steps):
            
            a = alphas[tid-1,:,:]
            alphas[tid,:,0] += a[:,0]
            alphas[tid,:,range(2,num_states,2)] += torch.logsumexp(a.unfold(1, 3, 2), dim=1)
            alphas[tid,:,range(1,num_states-1,2)] += torch.logsumexp(a.unfold(1, 2, 2), dim=1)
            
            # Vocab id is this character's index in the emission matrix
            # -1 because uses the prevous emission to calculate this time step's alpha

            # Adding log probabilities of the previous alphas, the transitions and the emission
            # Should be a self.num_states by self.num_states matrix
            # array[0,1] = log(alpha0) + log(p('e'|s0)) + log(p(s1|s0))
            # From s0 to s1 ^^^
            # Need to expand dims to make broadcasting work

            # Sum all of the probs in the columns, which means logsumexping all of the log probs

        # Final time step, single end state where all alphas are summed to get log-likelihood
        log_marginal_prob = torch.tensor(0.)

        for b in range(batch_size):
            sid = 2 * target_lens[b] - 1
            tid = pred_lens[b]
            log_marginal_prob+=alphas[tid,b,sid]
        
        return log_marginal_prob