"""
Multi-Endpoint API System Initialization Script
================================================

This script initializes the entire multi-endpoint API system:
1. Creates database tables
2. Loads endpoint configurations
3. Stores parameters
4. Generates and stores examples
5. Creates vector embeddings

Run this script once to set up the system, or run it again
to refresh the data if you've made changes to the configuration.
"""

import sys
import traceback
from query_store import QueryStore


def main():
    """Initialize the multi-endpoint API system"""
    
    print("=" * 70)
    print("Multi-Endpoint API System Initialization")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Initialize QueryStore
        print("📦 Step 1: Connecting to database...")
        query_store = QueryStore()
        print("✅ Connected successfully")
        print()
        
        # Step 2: Import configurations
        print("📋 Step 2: Loading configurations...")
        try:
            from api_endpoints_config import API_ENDPOINTS, get_all_endpoint_names
            from api_examples_generator import generate_api_examples, get_example_count_by_endpoint
            print(f"✅ Loaded {len(API_ENDPOINTS)} endpoint configurations")
            print(f"   Endpoints: {', '.join(get_all_endpoint_names())}")
        except ImportError as e:
            print(f"❌ Failed to import configurations: {e}")
            print("   Make sure api_endpoints_config.py and api_examples_generator.py are in the same directory")
            return False
        print()
        
        # Step 3: Create database tables
        print("🗄️  Step 3: Creating database tables...")
        success = query_store.initialize_api_tables_v2()
        if success:
            print("✅ Tables created successfully")
        else:
            print("⚠️  Tables may already exist (this is okay)")
        print()
        
        # Step 4: Store endpoints
        print("📍 Step 4: Storing API endpoints...")
        endpoint_count = query_store.store_api_endpoints(API_ENDPOINTS)
        print(f"✅ Stored {endpoint_count} endpoints")
        print()
        
        # Step 5: Store parameters for each endpoint
        print("⚙️  Step 5: Storing API parameters...")
        total_params = 0
        for endpoint_name, config in API_ENDPOINTS.items():
            parameters = config.get('parameters', {})
            if parameters:
                param_count = query_store.store_api_parameters_v2(endpoint_name, parameters)
                total_params += param_count
                print(f"   {endpoint_name}: {param_count} parameters")
        print(f"✅ Stored {total_params} total parameters")
        print()
        
        # Step 6: Generate and store examples
        print("📚 Step 6: Generating and storing examples...")
        examples = generate_api_examples()
        example_count = query_store.store_api_examples_v2(examples)
        print(f"✅ Stored {example_count} examples")
        
        # Show breakdown by endpoint
        print("\n   Examples by endpoint:")
        for endpoint, count in get_example_count_by_endpoint().items():
            print(f"   • {endpoint}: {count} examples")
        print()
        
        # Step 7: Verify setup
        print("🔍 Step 7: Verifying setup...")
        
        # Test endpoint search
        test_query = "show orders shipped today"
        endpoints = query_store.search_api_endpoints(test_query, top_k=1)
        if endpoints:
            endpoint_name, _, _, _, similarity = endpoints[0]
            print(f"✅ Endpoint search working (test query → {endpoint_name}, similarity: {similarity:.2f})")
        else:
            print("⚠️  Endpoint search returned no results")
        
        # Test parameter search
        params = query_store.search_api_parameters_v2(test_query, top_k=1)
        if params:
            _, param_name, _, _, _, similarity = params[0]
            print(f"✅ Parameter search working (test query → {param_name}, similarity: {similarity:.2f})")
        else:
            print("⚠️  Parameter search returned no results")
        
        # Test example search
        examples = query_store.search_api_examples_v2(test_query, top_k=1)
        if examples:
            nl, endpoint, _, _, similarity = examples[0]
            print(f"✅ Example search working (similarity: {similarity:.2f})")
        else:
            print("⚠️  Example search returned no results")
        
        print()
        print("=" * 70)
        print("✅ INITIALIZATION COMPLETE!")
        print("=" * 70)
        print()
        print("You can now use the MultiEndpointAPIAgent:")
        print()
        print("  from api_agent_multi_endpoint import MultiEndpointAPIAgent")
        print("  agent = MultiEndpointAPIAgent(query_store)")
        print('  result = agent.process_query("your query", {}, {}, silent_mode=False)')
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ INITIALIZATION FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print()
        return False


def refresh_data():
    """
    Refresh the data without recreating tables
    Useful when you've updated configurations or examples
    """
    print("=" * 70)
    print("Refreshing Multi-Endpoint API Data")
    print("=" * 70)
    print()
    
    try:
        from api_endpoints_config import API_ENDPOINTS
        from api_examples_generator import generate_api_examples, get_example_count_by_endpoint
        
        query_store = QueryStore()
        
        # Re-store endpoints (will update existing)
        print("📍 Refreshing endpoints...")
        query_store.store_api_endpoints(API_ENDPOINTS)
        
        # Re-store parameters
        print("⚙️  Refreshing parameters...")
        for endpoint_name, config in API_ENDPOINTS.items():
            parameters = config.get('parameters', {})
            if parameters:
                query_store.store_api_parameters_v2(endpoint_name, parameters)
        
        # Clear and re-store examples
        print("📚 Refreshing examples...")
        with query_store.conn.cursor() as cur:
            cur.execute("DELETE FROM api_examples_v2")
            query_store.conn.commit()
        
        examples = generate_api_examples()
        query_store.store_api_examples_v2(examples)
        
        print()
        print("✅ Data refresh complete!")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Refresh failed: {e}")
        traceback.print_exc()
        return False


def show_stats():
    """Show statistics about the stored data"""
    try:
        query_store = QueryStore()
        
        print()
        print("=" * 70)
        print("Multi-Endpoint API System Statistics")
        print("=" * 70)
        print()
        
        with query_store.conn.cursor() as cur:
            # Count endpoints
            cur.execute("SELECT COUNT(*) FROM api_endpoints")
            endpoint_count = cur.fetchone()[0]
            print(f"📍 Endpoints: {endpoint_count}")
            
            # Show endpoints
            cur.execute("SELECT endpoint_name FROM api_endpoints ORDER BY endpoint_name")
            endpoints = cur.fetchall()
            for (endpoint_name,) in endpoints:
                print(f"   • {endpoint_name}")
            print()
            
            # Count parameters
            cur.execute("SELECT COUNT(*) FROM api_parameters_v2")
            param_count = cur.fetchone()[0]
            print(f"⚙️  Total Parameters: {param_count}")
            
            # Parameters by endpoint
            cur.execute("""
                SELECT endpoint_name, COUNT(*) 
                FROM api_parameters_v2 
                GROUP BY endpoint_name 
                ORDER BY endpoint_name
            """)
            param_breakdown = cur.fetchall()
            for endpoint_name, count in param_breakdown:
                print(f"   • {endpoint_name}: {count} parameters")
            print()
            
            # Count examples
            cur.execute("SELECT COUNT(*) FROM api_examples_v2")
            example_count = cur.fetchone()[0]
            print(f"📚 Total Examples: {example_count}")
            
            # Examples by endpoint
            cur.execute("""
                SELECT endpoint_name, COUNT(*) 
                FROM api_examples_v2 
                GROUP BY endpoint_name 
                ORDER BY endpoint_name
            """)
            example_breakdown = cur.fetchall()
            for endpoint_name, count in example_breakdown:
                print(f"   • {endpoint_name}: {count} examples")
            print()
            
            query_store.conn.commit()
        
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"\n❌ Failed to get statistics: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "refresh":
            refresh_data()
        elif command == "stats":
            show_stats()
        elif command == "help":
            print()
            print("Multi-Endpoint API System Initialization")
            print()
            print("Usage:")
            print("  python initialize_api_system.py          # Full initialization")
            print("  python initialize_api_system.py refresh  # Refresh data only")
            print("  python initialize_api_system.py stats    # Show statistics")
            print("  python initialize_api_system.py help     # Show this help")
            print()
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' to see available commands")
    else:
        # Run full initialization
        success = main()
        sys.exit(0 if success else 1)