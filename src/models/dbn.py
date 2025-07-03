class DbN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8]):
        super(DbN, self).__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                    bias=False,
                )
                for rate in dilation_rates
            ]
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        dilated_features = [conv(x) for conv in self.convs]
        x = sum(dilated_features)  # Summing outputs from different dilated convolutions
        x = self.batch_norm(x)
        return self.relu(x)