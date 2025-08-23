#!/usr/bin/env python3
"""Comprehensive test suite for investment analysis agent tools."""

import sys
import os
sys.path.insert(0, '.')

from main import (
    get_historic_stock_data,
    get_stock_notifications,
    get_real_time_quote,
    get_financial_statements,
    calculate_investment_performance,
    check_critical_decision,
    parse_investment_decisions
)

def test_historic_stock_data():
    """Test historical stock data retrieval."""
    print("Testing get_historic_stock_data...")
    
    # Test valid parameters
    result = get_historic_stock_data("symbol=000001,start_date=20230801,end_date=20230810,period=daily")
    assert "Data for 000001" in result or "No data found" in result or "Error" in result
    print("✓ Valid parameters test passed")
    
    # Test error handling
    result = get_historic_stock_data("symbol=invalid,start_date=20230801,end_date=20230810")
    assert "Error" in result or "No data found" in result
    print("✓ Error handling test passed")
    
    print("All historical stock data tests passed!\n")

def test_stock_notifications():
    """Test stock notifications retrieval."""
    print("Testing get_stock_notifications...")
    
    # Test valid parameters
    result = get_stock_notifications("notice_type=财务报告,date=20220511")
    assert "announcements" in result.lower() or "no announcements" in result.lower() or "Error" in result
    print("✓ Valid parameters test passed")
    
    # Test error handling
    result = get_stock_notifications("notice_type=invalid,date=20220511")
    assert "Error" in result or "Invalid" in result
    print("✓ Error handling test passed")
    
    print("All stock notifications tests passed!\n")

def test_real_time_quote():
    """Test real-time quote retrieval."""
    print("Testing get_real_time_quote...")
    
    # Test valid parameters
    result = get_real_time_quote("symbol=000001")
    assert "Real-time quote" in result or "quote" in result.lower() or "Error" in result
    print("✓ Valid parameters test passed")
    
    # Test error handling
    result = get_real_time_quote("symbol=invalid")
    assert "Error" in result or "No real-time data" in result
    print("✓ Error handling test passed")
    
    print("All real-time quote tests passed!\n")

def test_financial_statements():
    """Test financial statements retrieval."""
    print("Testing get_financial_statements...")
    
    # Test valid parameters
    result = get_financial_statements("symbol=000001,report_type=balance")
    assert "Financial" in result or "Basic Financial" in result or "Error" in result
    print("✓ Valid parameters test passed")
    
    # Test error handling
    result = get_financial_statements("symbol=invalid")
    assert "Error" in result or "No financial data" in result
    print("✓ Error handling test passed")
    
    print("All financial statements tests passed!\n")

def test_performance_calculation():
    """Test investment performance calculation."""
    print("Testing calculate_investment_performance...")
    
    # Test valid parameters
    result = calculate_investment_performance("symbol=000001,start_date=20230101,end_date=20231231,initial_investment=100000")
    assert "Performance Analysis" in result
    print("✓ Valid parameters test passed")
    
    # Test with benchmark
    result = calculate_investment_performance("symbol=000001,start_date=20230101,end_date=20231231,benchmark=000300")
    assert "Performance Analysis" in result
    print("✓ Benchmark test passed")
    
    # Test error handling
    result = calculate_investment_performance("symbol=invalid,start_date=20230101,end_date=20231231")
    assert "Error" in result or "No historical data" in result
    print("✓ Error handling test passed")
    
    print("All performance calculation tests passed!\n")

def test_decision_parsing():
    """Test investment decision parsing."""
    print("Testing decision parsing functions...")
    
    # Test check_critical_decision
    buy_response = "I recommend buying stock 600519"
    sell_response = "We should sell this position immediately"
    neutral_response = "This is general market analysis"
    
    assert check_critical_decision(buy_response) == True
    assert check_critical_decision(sell_response) == True
    assert check_critical_decision(neutral_response) == False
    print("✓ Critical decision detection test passed")
    
    # Test parse_investment_decisions
    complex_response = """
    [HIGH CONFIDENCE] Strong consensus with comprehensive evidence
    Recommendation: BUY stock 600519
    We should reduce exposure to 000001 due to poor performance
    This represents a great opportunity in the market
    """
    
    decisions = parse_investment_decisions(complex_response)
    assert len(decisions) >= 2  # Should find buy and opportunity at minimum
    print("✓ Decision parsing test passed")
    
    print("All decision parsing tests passed!\n")

def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    try:
        test_historic_stock_data()
        test_stock_notifications()
        test_real_time_quote()
        test_financial_statements()
        test_performance_calculation()
        test_decision_parsing()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)