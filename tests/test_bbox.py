import pytest
from tests.conftest import DummyPlayer

def bbox_center(b):    
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)
def bbox_width(b):     
    return b[2] - b[0]
def bbox_height(b):    
    return b[3] - b[1]
def bbox_bottom_center(b): 
    return ((b[0]+b[2])/2, b[3])

class TestBbox:
    def test_center(self):           
        assert bbox_center([0,0,4,6]) == (2.0, 3.0)
        
    def test_width(self):            
        assert bbox_width([10,0,30,50]) == 20
        
    def test_height(self):           
        assert bbox_height([10,0,30,50]) == 50
        
    def test_bottom_center(self):    
        assert bbox_bottom_center([0,0,10,20]) == (5.0, 20)
        
    def test_zero_area(self):        
        assert bbox_width([5,5,5,5]) == bbox_height([5,5,5,5]) == 0
        
    def test_negative(self):         
        assert bbox_center([-10,-5,10,5]) == (0.0, 0.0)
        
    def test_bottom_below_center(self):
        cx, cy = bbox_center([2,3,6,11])
        bx, by = bbox_bottom_center([2,3,6,11])
        assert cx == 4 and by > cy
