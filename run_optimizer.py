def run_optimizer(opt, steps=200):

    x = torch.tensor([-2.0, 2.0], requires_grad=True)

    trajectory = []

    for _ in range(steps):

        loss = rosenbrock(x)

        loss.backward()

        grad = x.grad.detach()

        x = opt.step(x.detach(), grad)

        x.requires_grad = True

        trajectory.append(x.detach().numpy())

    return trajectory
