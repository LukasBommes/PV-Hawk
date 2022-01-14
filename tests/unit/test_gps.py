import unittest
import numpy as np
from hypothesis import given
import hypothesis.strategies as some
import hypothesis.extra.numpy as some_np

from extractor.gps import gps_to_ltp, gps_from_ltp, \
    interpolate_gps


class TestGps(unittest.TestCase):
    
    @given(
        some_np.arrays(
            dtype=np.float, 
            shape=some.tuples(
                some.integers(min_value=1, max_value=5), # 1..5 rows
                some.integers(min_value=3, max_value=3)  # 3 columns
            ), 
            elements=some.floats(-90, 90)
        )
    )
    def test_gps_to_ltp_consistency(self, gps):
        
        gps_ltp, origin = gps_to_ltp(gps)
        gps_recovered = gps_from_ltp(gps_ltp, origin)

        self.assertTrue(
            np.allclose(
                (gps[0, 1], gps[0, 0], gps[0, 2]), 
                origin
            )
        )
        self.assertTrue(
            np.allclose(
                gps, 
                gps_recovered
            )
        )

    def test_gps_interpolation(self):
        gps = np.array([
            [10., 10., 10.], 
            [10., 10., 10.], 
            [10., 10., 10.],
            [10., 10., 10.],
            [12., 15.,  8.],
            [12., 15.,  8.],
            [12., 15.,  8.],
            [12., 15.,  8.],
            [15., 20.,  5.],
            [15., 20.,  5.],
            [15., 20.,  5.],
            [15., 20.,  5.],
            [20., 25., 10.],
            [20., 25., 10.],
            [20., 25., 10.],
            [20., 25., 10.]
        ])

        gps_interpolated_gt = np.array([
            [10.  , 10.  , 10.  ],
            [10.5 , 11.25,  9.5 ],
            [11.  , 12.5 ,  9.  ],
            [11.5 , 13.75,  8.5 ],
            [12.  , 15.  ,  8.  ],
            [12.75, 16.25,  7.25],
            [13.5 , 17.5 ,  6.5 ],
            [14.25, 18.75,  5.75],
            [15.  , 20.  ,  5.  ],
            [16.25, 21.25,  6.25],
            [17.5 , 22.5 ,  7.5 ],
            [18.75, 23.75,  8.75],
            [20.  , 25.  , 10.  ],
            [20.  , 25.  , 10.  ],
            [20.  , 25.  , 10.  ],
            [20.  , 25.  , 10.  ]
        ])
        
        gps_interpolated = interpolate_gps(gps)        
        self.assertTrue(
            np.allclose(
                gps_interpolated_gt, 
                gps_interpolated
            )
        )