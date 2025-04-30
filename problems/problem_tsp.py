from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle

class TSP(object):

    NAME = 'tsp'
    
    def __init__(self, p_size=20, with_assert = False):

        self.size = p_size          # the number of nodes in tsp 
        self.do_assert = with_assert
        self.optimal_tours = None   # Will store optimal tours if provided
        print(f'TSP with {self.size} nodes.')
    
    def load_optimal_tours(self, file_path):
        """Load pre-computed optimal tours from file"""
        if file_path is None or not os.path.exists(file_path):
            print(f"Warning: Optimal tours file {file_path} not found.")
            return False
            
        try:
            with open(file_path, 'rb') as f:
                self.optimal_tours = pickle.load(f)
            print(f"Loaded {len(self.optimal_tours)} optimal tours.")
            return True
        except Exception as e:
            print(f"Error loading optimal tours: {e}")
            return False
    
    def calculate_edge_overlap(self, tours, optimal_tours):
        """
        Calculate the edge overlap between current tours and optimal tours
        
        Args:
            tours: (batch_size, graph_size) current tours
            optimal_tours: (batch_size, graph_size) optimal tours
            
        Returns:
            overlap: (batch_size) number of shared edges
        """
        batch_size, graph_size = tours.size()
        device = tours.device
        
        # Initialize overlap count
        overlap = torch.zeros(batch_size, device=device)
        
        # Convert to numpy for processing
        tours_np = tours.cpu().numpy()
        optimal_tours_np = optimal_tours.cpu().numpy()
        
        for i in range(batch_size):
            # Get edges in current tour (include the last edge connecting back to start)
            tour_edges = set()
            for j in range(graph_size):
                n1, n2 = tours_np[i][j], tours_np[i][(j+1) % graph_size]
                # Undirected edge - store in canonical form (smaller node first)
                tour_edges.add((min(n1, n2), max(n1, n2)))
            
            # Get edges in optimal tour
            opt_edges = set()
            for j in range(graph_size):
                n1, n2 = optimal_tours_np[i][j], optimal_tours_np[i][(j+1) % graph_size]
                opt_edges.add((min(n1, n2), max(n1, n2)))
            
            # Count overlap
            overlap[i] = len(tour_edges.intersection(opt_edges))
        
        return overlap
    
    def calculate_broken_optimal_edges(self, tours, exchange, optimal_tours):
        """
        Calculate how many optimal edges would be broken by an exchange
        
        Args:
            tours: (batch_size, graph_size) current tours
            exchange: (batch_size, 2) nodes to exchange
            optimal_tours: (batch_size, graph_size) optimal tours
            
        Returns:
            broken_count: (batch_size) number of optimal edges that would be broken
        """
        batch_size = tours.size(0)
        device = tours.device
        
        # Initialize broken count
        broken_count = torch.zeros(batch_size, device=device)
        
        # Convert to numpy for processing
        tours_np = tours.cpu().numpy()
        exchange_np = exchange.cpu().numpy()
        optimal_tours_np = optimal_tours.cpu().numpy()
        
        for i in range(batch_size):
            # Get edges in optimal tour
            opt_edges = set()
            for j in range(self.size):
                n1, n2 = optimal_tours_np[i][j], optimal_tours_np[i][(j+1) % self.size]
                opt_edges.add((min(n1, n2), max(n1, n2)))
            
            # Find which edges would be removed by this exchange
            loc_of_first = np.where(tours_np[i] == exchange_np[i][0])[0][0]
            loc_of_second = np.where(tours_np[i] == exchange_np[i][1])[0][0]
            
            # Adjustment for 2-opt
            if loc_of_first > loc_of_second:
                loc_of_first, loc_of_second = loc_of_second, loc_of_first
            
            # Edges that would be broken (one before first, one after second, and connection between)
            broken_edges = set()
            
            # Edge before first node
            n1, n2 = tours_np[i][(loc_of_first-1) % self.size], tours_np[i][loc_of_first]
            broken_edges.add((min(n1, n2), max(n1, n2)))
            
            # Edge after second node
            n1, n2 = tours_np[i][loc_of_second], tours_np[i][(loc_of_second+1) % self.size]
            broken_edges.add((min(n1, n2), max(n1, n2)))
            
            # Count how many of these are optimal edges
            broken_count[i] = len(broken_edges.intersection(opt_edges))
        
        return broken_count
    
    def step(self, rec, exchange):
        
        device = rec.device
        
        exchange_num = exchange.clone().cpu().numpy()
        rec_num = rec.clone().cpu().numpy()
         
        for i in range(rec.size()[0]):
            
            loc_of_first = np.where(rec_num[i] == exchange_num[i][0])[0][0]
            loc_of_second = np.where(rec_num[i] == exchange_num[i][1])[0][0]
            
            if( loc_of_first < loc_of_second):
                rec_num[i][loc_of_first:loc_of_second+1] = np.flip(
                        rec_num[i][loc_of_first:loc_of_second+1])
            else:
                temp = rec_num[i][loc_of_first]
                rec_num[i][loc_of_first] = rec_num[i][loc_of_second]
                rec_num[i][loc_of_second] = temp
                
        return torch.tensor(rec_num, device = device)
            
    
    def get_costs(self, dataset, rec):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param rec: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        if self.do_assert:
            assert (
                torch.arange(rec.size(1), out=rec.data.new()).view(1, -1).expand_as(rec) ==
                rec.data.sort(1)[0]
            ).all(), "Invalid tour"
        
        # Gather dataset in order of tour
        d = dataset.gather(1, rec.long().unsqueeze(-1).expand_as(dataset))
        length =  (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
        
        return length
    
    
    def get_swap_mask(self, rec):
        _, graph_size = rec.size()
        return torch.eye(graph_size).view(1,graph_size,graph_size)

    def get_initial_solutions(self, methods, batch):
        
        def seq_solutions(batch_size):
            graph_size = self.size
            solution = torch.linspace(0, graph_size-1, steps=graph_size)
            return solution.expand(batch_size,graph_size).clone()
        
        batch_size = len(batch)
        
        if methods == 'seq':
            return seq_solutions(batch_size)
            
    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)   



class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=20, num_samples=10000):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in data[:num_samples]]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
