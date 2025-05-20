
## Decode_head


### `DepthwiseSeparableASPPHead`
```python
    class DepthwiseSeparableASPPHead(ASPPHead):
        """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
        Segmentation.

        This head is the implementation of `DeepLabV3+
        <https://arxiv.org/abs/1802.02611>`_.

        Args:
            c1_in_channels (int): The input channels of c1 decoder. If is 0,
                the no decoder will be used.
            c1_channels (int): The intermediate channels of c1 decoder.
        """

        def __init__(self, c1_in_channels, c1_channels, **kwargs):
            super().__init__(**kwargs)
            assert c1_in_channels >= 0
            self.aspp_modules = DepthwiseSeparableASPPModule(
                dilations=self.dilations,
                in_channels=self.in_channels,
                channels=self.channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            if c1_in_channels > 0:
                self.c1_bottleneck = ConvModule(
                    c1_in_channels,
                    c1_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            else:
                self.c1_bottleneck = None
            self.sep_bottleneck = nn.Sequential(
                DepthwiseSeparableConvModule(
                    self.channels + c1_channels,
                    self.channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                DepthwiseSeparableConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
```

### `ASPPHead`
```python
    class ASPPHead(BaseDecodeHead):
        """Rethinking Atrous Convolution for Semantic Image Segmentation.

        This head is the implementation of `DeepLabV3
        <https://arxiv.org/abs/1706.05587>`_.

        Args:
            dilations (tuple[int]): Dilation rates for ASPP module.
                Default: (1, 6, 12, 18).
        """

        def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
            super().__init__(**kwargs)
            assert isinstance(dilations, (list, tuple))
            self.dilations = dilations
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.aspp_modules = ASPPModule(
                dilations,
                self.in_channels,
                self.channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.bottleneck = ConvModule(
                (len(dilations) + 1) * self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
```



```python
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output
```

#### `_transform_inputs`

* inputs: feature list from backbone and it is transformed by the below

```python
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
```

* Thus, `input_transform` determines how to use the features from the backbone.

* When it is `None`, it just selects the last feature of the list of features from backbone. For example, 
    * inputs: [torch.Size([2, 256, 192, 280]), 
                torch.Size([2, 512, 96, 140]),
                torch.Size([2, 1024, 96, 140]),
                torch.Size([2, 2048, 96, 140])
        ]
    * transformed inputs: torch.Size([2, 2048, 96, 140])

* The number of channel of transformed inputs determines the number of input channel of decoder 

<details>
    <summary> ResNet Backbone </summary>


#### `image_pool`

```python
    Sequential(
                (0): AdaptiveAvgPool2d(output_size=1)
                (1): ConvModule(
                            (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                            (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                            (activate): ReLU(inplace=True)
                    )
            )
```

#### `aspp_modules`

The `aspp_outs` has 5 features where each of them has (bs, 512, h', w') shape and after `concat`, it is (bs, 512*5, w', h').


```python
    DepthwiseSeparableASPPModule(
    (0): ConvModule(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
    )
    (1): DepthwiseSeparableConvModule(
        (depthwise_conv): ConvModule(
        (conv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12), groups=2048, bias=False)
        (bn): _BatchNormXd(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
        )
        (pointwise_conv): ConvModule(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
        )
    )
    (2): DepthwiseSeparableConvModule(
        (depthwise_conv): ConvModule(
        (conv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24), groups=2048, bias=False)
        (bn): _BatchNormXd(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
        )
        (pointwise_conv): ConvModule(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
        )
    )
    (3): DepthwiseSeparableConvModule(
        (depthwise_conv): ConvModule(
        (conv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(36, 36), dilation=(36, 36), groups=2048, bias=False)
        (bn): _BatchNormXd(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
        )
        (pointwise_conv): ConvModule(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
        )
    )
    )
```


#### `bottleneck`

```python
    ConvModule(
            (conv): Conv2d(2560, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
    )
```

</details>

<details>
    <summary> EfficientNetV2 </summary>

#### `bottleneck`
```python
    ConvModule(
            (conv): Conv2d(2560, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
    )
```

#### `c1_bottleneck`

```python
    ConvModule(
            (conv): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): _BatchNormXd(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
    )
```


#### `sep_bottleneck`

```python
    Sequential(
            (0): DepthwiseSeparableConvModule(
                        (depthwise_conv): ConvModule(
                        (conv): Conv2d(560, 560, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=560, bias=False)
                        (bn): _BatchNormXd(560, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (activate): ReLU(inplace=True)
                        )
                        (pointwise_conv): ConvModule(
                        (conv): Conv2d(560, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (activate): ReLU(inplace=True)
                        )
            )
            (1): DepthwiseSeparableConvModule(
                        (depthwise_conv): ConvModule(
                        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
                        (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (activate): ReLU(inplace=True)
                        )
                        (pointwise_conv): ConvModule(
                        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (bn): _BatchNormXd(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (activate): ReLU(inplace=True)
                        )
            )
    )
```
</details>