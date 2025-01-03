# Elementwise Add

This folder consists some kernels to calculate elementwise add.

## command
```
python elementwise.py
```

## output on RTX3080
```
-------------------------------------------------------------------------------------
                                        S=1024, K=1024
           out_f32: [-1.68851185, 0.05288404], time:0.02092242ms
         out_f32x4: [-1.68851185, 0.05288404], time:0.02046704ms
        out_f32_th: [-1.68851185, 0.05288404], time:0.02171993ms
-------------------------------------------------------------------------------------
           out_f16: [-1.68847656, 0.05273438], time:0.01427388ms
      out_f16_hadd: [-1.68847656, 0.05273438], time:0.01435995ms
         out_f16x2: [-1.68847656, 0.05273438], time:0.01025605ms
         out_f16x8: [-1.68847656, 0.05273438], time:0.00972676ms
     out_f16x8pack: [-1.68847656, 0.05273438], time:0.00978971ms
        out_f16_th: [-1.68847656, 0.05273438], time:0.00998950ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=1024, K=2048
           out_f32: [-0.06172496, 0.16290754], time:0.04371834ms
         out_f32x4: [-0.06172496, 0.16290754], time:0.04070425ms
        out_f32_th: [-0.06172496, 0.16290754], time:0.03999639ms
-------------------------------------------------------------------------------------
           out_f16: [-0.06201172, 0.16308594], time:0.02105165ms
      out_f16_hadd: [-0.06201172, 0.16308594], time:0.02249050ms
         out_f16x2: [-0.06201172, 0.16308594], time:0.02090025ms
         out_f16x8: [-0.06201172, 0.16308594], time:0.02083254ms
     out_f16x8pack: [-0.06201172, 0.16308594], time:0.02031040ms
        out_f16_th: [-0.06201172, 0.16308594], time:0.02181387ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=1024, K=4096
           out_f32: [-0.77569801, -0.78907281], time:0.07828951ms
         out_f32x4: [-0.77569801, -0.78907281], time:0.07824922ms
        out_f32_th: [-0.77569801, -0.78907281], time:0.07861328ms
-------------------------------------------------------------------------------------
           out_f16: [-0.77539062, -0.78857422], time:0.04153609ms
      out_f16_hadd: [-0.77539062, -0.78857422], time:0.04186249ms
         out_f16x2: [-0.77539062, -0.78857422], time:0.04257321ms
         out_f16x8: [-0.77539062, -0.78857422], time:0.04090500ms
     out_f16x8pack: [-0.77539062, -0.78857422], time:0.04096556ms
        out_f16_th: [-0.77539062, -0.78857422], time:0.04319072ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=2048, K=1024
           out_f32: [0.61982077, 0.7278533], time:0.04105568ms
         out_f32x4: [0.61982077, 0.7278533], time:0.04134274ms
        out_f32_th: [0.61982077, 0.7278533], time:0.04009485ms
-------------------------------------------------------------------------------------
           out_f16: [0.61962891, 0.72753906], time:0.02469444ms
      out_f16_hadd: [0.61962891, 0.72753906], time:0.02417684ms
         out_f16x2: [0.61962891, 0.72753906], time:0.02215385ms
         out_f16x8: [0.61962891, 0.72753906], time:0.02086878ms
     out_f16x8pack: [0.61962891, 0.72753906], time:0.02092171ms
        out_f16_th: [0.61962891, 0.72753906], time:0.02179790ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=2048, K=2048
           out_f32: [-0.66956168, -1.10302222], time:0.07912326ms
         out_f32x4: [-0.66956168, -1.10302222], time:0.08180285ms
        out_f32_th: [-0.66956168, -1.10302222], time:0.07834387ms
-------------------------------------------------------------------------------------
           out_f16: [-0.66992188, -1.10351562], time:0.04169536ms
      out_f16_hadd: [-0.66992188, -1.10351562], time:0.04110050ms
         out_f16x2: [-0.66992188, -1.10351562], time:0.04061580ms
         out_f16x8: [-0.66992188, -1.10351562], time:0.04191446ms
     out_f16x8pack: [-0.66992188, -1.10351562], time:0.04104733ms
        out_f16_th: [-0.66992188, -1.10351562], time:0.04155564ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=2048, K=4096
           out_f32: [-2.12097859, 2.16298842], time:0.15040088ms
         out_f32x4: [-2.12097859, 2.16298842], time:0.15223312ms
        out_f32_th: [-2.12097859, 2.16298842], time:0.15228486ms
-------------------------------------------------------------------------------------
           out_f16: [-2.12109375, 2.1640625], time:0.07879543ms
      out_f16_hadd: [-2.12109375, 2.1640625], time:0.07837367ms
         out_f16x2: [-2.12109375, 2.1640625], time:0.07944632ms
         out_f16x8: [-2.12109375, 2.1640625], time:0.07833552ms
     out_f16x8pack: [-2.12109375, 2.1640625], time:0.07873607ms
        out_f16_th: [-2.12109375, 2.1640625], time:0.07908297ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=4096, K=1024
           out_f32: [2.54709864, -0.92635918], time:0.07532334ms
         out_f32x4: [2.54709864, -0.92635918], time:0.07813811ms
        out_f32_th: [2.54709864, -0.92635918], time:0.07781029ms
-------------------------------------------------------------------------------------
           out_f16: [2.546875, -0.92675781], time:0.04711437ms
      out_f16_hadd: [2.546875, -0.92675781], time:0.04804993ms
         out_f16x2: [2.546875, -0.92675781], time:0.04126096ms
         out_f16x8: [2.546875, -0.92675781], time:0.04027653ms
     out_f16x8pack: [2.546875, -0.92675781], time:0.03951073ms
        out_f16_th: [2.546875, -0.92675781], time:0.04257011ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=4096, K=2048
           out_f32: [-0.2261367, 0.16115648], time:0.15071726ms
         out_f32x4: [-0.2261367, 0.16115648], time:0.15474248ms
        out_f32_th: [-0.2261367, 0.16115648], time:0.15458488ms
-------------------------------------------------------------------------------------
           out_f16: [-0.22607422, 0.16113281], time:0.08032393ms
      out_f16_hadd: [-0.22607422, 0.16113281], time:0.08263326ms
         out_f16x2: [-0.22607422, 0.16113281], time:0.07741237ms
         out_f16x8: [-0.22607422, 0.16113281], time:0.07768464ms
     out_f16x8pack: [-0.22607422, 0.16113281], time:0.07813454ms
        out_f16_th: [-0.22607422, 0.16113281], time:0.07878470ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=4096, K=4096
           out_f32: [0.31718218, -0.48734733], time:0.29643011ms
         out_f32x4: [0.31718218, -0.48734733], time:0.29988241ms
        out_f32_th: [0.31718218, -0.48734733], time:0.29853272ms
-------------------------------------------------------------------------------------
           out_f16: [0.31713867, -0.48754883], time:0.15268493ms
      out_f16_hadd: [0.31713867, -0.48754883], time:0.15432501ms
         out_f16x2: [0.31713867, -0.48754883], time:0.15268493ms
         out_f16x8: [0.31713867, -0.48754883], time:0.15437365ms
     out_f16x8pack: [0.31713867, -0.48754883], time:0.15318751ms
        out_f16_th: [0.31713867, -0.48754883], time:0.15402699ms
-------------------------------------------------------------------------------------
```