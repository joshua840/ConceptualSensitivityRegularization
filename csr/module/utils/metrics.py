class Metrics:
    def get_binary_accuracy(self, logit, y, threshold=0):
        y_hat = (logit >= threshold).long().reshape(-1)
        y = y.reshape(-1)
        return (y_hat == y).long().sum().item() * 1.0 / len(y_hat)

    def get_accuracy(self, y_hat, y, criterion):
        if criterion == "ce":
            return (y_hat.argmax(dim=1) == y).sum().item() * 1.0 / len(y)
        elif criterion == "bce":
            return self.get_binary_accuracy(y_hat, y)
