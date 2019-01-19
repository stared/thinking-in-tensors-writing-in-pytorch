class Factorize(nn.Module):
    
    def __init__(self, factors=2):
        super().__init__()
        self.A = Parameter(torch.randn(9, factors))
        self.B = Parameter(torch.randn(factors, 12))
    
    def forward(self):
        output = self.A.matmul(self.B)
        return output
    