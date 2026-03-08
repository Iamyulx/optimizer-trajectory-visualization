class Adam:

    def __init__(self, lr=0.05, betas=(0.9,0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps

        self.m = torch.zeros(2)
        self.v = torch.zeros(2)
        self.t = 0

    def step(self, x, grad):

        self.t += 1

        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*(grad**2)

        m_hat = self.m/(1-self.b1**self.t)
        v_hat = self.v/(1-self.b2**self.t)

        return x - self.lr*m_hat/(torch.sqrt(v_hat)+self.eps)
