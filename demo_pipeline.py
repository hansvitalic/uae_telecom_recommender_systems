#!/usr/bin/env python3
"""
Simple RMS Pipeline Demonstration
Runs the core RMS modules and ML feature engineering to generate key outputs
"""

import sys
import os
sys.path.append('src')

from rms_pipeline.risk_identification import RiskIdentifier
from rms_pipeline.qualitative_analysis import QualitativeAnalyzer
from rms_pipeline.quantitative_analysis import QuantitativeAnalyzer
from rms_pipeline.response_planning import ResponsePlanner
from rms_pipeline.monitoring import RiskMonitor
from rms_pipeline.documentation import RMSDocumentationGenerator
from ml_models.feature_engineering import FeatureEngineer

import pandas as pd
import json
from datetime import datetime
import config

def main():
    """Run simplified RMS pipeline demonstration"""
    print("🚀 UAE Telecom RMS Pipeline Demonstration")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Load risk data
    print("📁 Loading risk register data...")
    risks_df = pd.read_csv(config.RISK_REGISTER_CSV)
    print(f"✓ Loaded {len(risks_df)} risks from register")
    
    # Run RMS Pipeline Components
    results = {}
    
    print("\n🔍 RISK IDENTIFICATION")
    print("-" * 40)
    identifier = RiskIdentifier()
    identification_result = identifier.generate_identification_report()
    results['identification'] = identification_result
    print(f"✓ Identified {identification_result['identification_summary']['total_risks_identified']} risks")
    print(f"✓ Data quality score: {identification_result['identification_summary']['data_quality_score']}")
    
    print("\n📊 QUALITATIVE ANALYSIS")
    print("-" * 40)
    qualitative_analyzer = QualitativeAnalyzer(risks_df)
    qualitative_result = qualitative_analyzer.generate_qualitative_report()
    results['qualitative'] = qualitative_result
    print(f"✓ Analyzed {qualitative_result['analysis_summary']['total_risks_analyzed']} risks")
    print(f"✓ Critical risks: {qualitative_result['analysis_summary']['critical_risks']}")
    print(f"✓ High priority risks: {qualitative_result['analysis_summary']['high_priority_risks']}")
    
    print("\n📈 QUANTITATIVE ANALYSIS")
    print("-" * 40)
    quantitative_analyzer = QuantitativeAnalyzer(risks_df)
    quantitative_result = quantitative_analyzer.generate_quantitative_report()
    results['quantitative'] = quantitative_result
    print(f"✓ Total risk exposure: ${quantitative_result['analysis_summary']['total_risk_exposure']:,.2f}")
    print(f"✓ Value at Risk (95%): ${quantitative_result['analysis_summary']['value_at_risk_95']:,.2f}")
    
    print("\n🎯 RESPONSE PLANNING")
    print("-" * 40)
    response_planner = ResponsePlanner(risks_df, results)
    response_result = response_planner.generate_response_report()
    results['response'] = response_result
    print(f"✓ Planned responses for {response_result['planning_summary']['total_risks_planned']} risks")
    print(f"✓ Immediate responses needed: {response_result['planning_summary']['immediate_responses_needed']}")
    
    print("\n👁️ RISK MONITORING")
    print("-" * 40)
    monitor = RiskMonitor(risks_df)
    monitoring_result = monitor.generate_monitoring_report()
    results['monitoring'] = monitoring_result
    print(f"✓ Monitoring {monitoring_result['monitoring_summary']['total_risks_monitored']} risks")
    print(f"✓ Active alerts: {monitoring_result['monitoring_summary']['active_alerts']}")
    
    print("\n📋 DOCUMENTATION")
    print("-" * 40)
    doc_generator = RMSDocumentationGenerator(risks_df, results)
    doc_result = doc_generator.generate_documentation_report()
    results['documentation'] = doc_result
    print(f"✓ Generated comprehensive documentation")
    
    print("\n🔧 ML FEATURE ENGINEERING")
    print("-" * 40)
    feature_engineer = FeatureEngineer(risks_df)
    ml_df, metadata = feature_engineer.prepare_ml_dataset()
    dataset_path = feature_engineer.export_ml_dataset()
    results['ml_features'] = {
        'dataset_path': dataset_path,
        'metadata': metadata
    }
    print(f"✓ Generated {metadata['total_features']} features")
    print(f"✓ Dataset shape: {metadata['data_shape']}")
    print(f"✓ ML dataset exported to: {dataset_path}")
    
    # Export processed RMS data
    print("\n📤 EXPORTING RMS PROCESSED DATA")
    print("-" * 40)
    
    # Create enhanced risk register
    enhanced_df = risks_df.copy()
    
    # Add priority levels from qualitative analysis
    if 'prioritized_risks' in qualitative_result:
        priority_map = {
            risk['Risk ID']: {
                'Enhanced_Score': risk['Enhanced Score'],
                'Priority_Level': risk['Priority Level']
            }
            for risk in qualitative_result['prioritized_risks']
        }
        
        enhanced_df['Enhanced_Risk_Score'] = enhanced_df['Risk ID'].map(
            lambda x: priority_map.get(x, {}).get('Enhanced_Score', 0)
        )
        enhanced_df['Priority_Level'] = enhanced_df['Risk ID'].map(
            lambda x: priority_map.get(x, {}).get('Priority_Level', 'Unknown')
        )
    
    # Add exposure data from quantitative analysis
    if 'exposure_data' in quantitative_result.get('risk_exposure', {}):
        exposure_df = quantitative_result['risk_exposure']['exposure_data']
        enhanced_df = enhanced_df.merge(
            exposure_df[['Risk ID', 'Probability %', 'Impact Cost', 'Expected Monetary Value']],
            on='Risk ID',
            how='left'
        )
    
    # Add processing metadata
    enhanced_df['RMS_Processing_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    enhanced_df['RMS_Pipeline_Version'] = '1.0'
    
    # Export enhanced data
    rms_output_path = config.RMS_OUTPUT_FILE
    enhanced_df.to_csv(rms_output_path, index=False)
    print(f"✓ RMS processed data exported to: {rms_output_path}")
    
    # Generate execution summary
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    summary = {
        "execution_metadata": {
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "execution_duration": str(execution_time),
            "total_risks_processed": len(risks_df)
        },
        "rms_pipeline_results": {
            "risks_identified": identification_result['identification_summary']['total_risks_identified'],
            "critical_risks": qualitative_result['analysis_summary']['critical_risks'],
            "total_risk_exposure": quantitative_result['analysis_summary']['total_risk_exposure'],
            "immediate_responses_needed": response_result['planning_summary']['immediate_responses_needed'],
            "active_alerts": monitoring_result['monitoring_summary']['active_alerts']
        },
        "ml_analysis_results": {
            "features_generated": metadata['total_features'],
            "dataset_shape": metadata['data_shape'],
            "ml_dataset_path": str(dataset_path)
        },
        "deliverables": {
            "rms_processed_data": str(rms_output_path),
            "ml_ready_dataset": str(dataset_path),
            "comprehensive_documentation": str(config.OUTPUT_DIR / "comprehensive_rms_documentation.json")
        },
        "key_insights": [
            f"Identified {identification_result['identification_summary']['critical_risks_count']} critical risks requiring immediate attention",
            f"Total financial risk exposure estimated at ${quantitative_result['analysis_summary']['total_risk_exposure']:,.2f}",
            f"Generated {metadata['total_features']} features for ML-based risk prediction",
            f"{response_result['planning_summary']['immediate_responses_needed']} risks require immediate response implementation"
        ]
    }
    
    # Export summary
    summary_path = config.OUTPUT_DIR / "pipeline_demonstration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n🎉 PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"✅ Execution time: {execution_time}")
    print(f"✅ Total risks processed: {len(risks_df)}")
    print(f"✅ Critical risks identified: {identification_result['identification_summary']['critical_risks_count']}")
    print(f"✅ ML features generated: {metadata['total_features']}")
    print(f"✅ Summary exported to: {summary_path}")
    
    print("\n📋 Key Deliverables:")
    for name, path in summary['deliverables'].items():
        print(f"  • {name}: {path}")
    
    return summary

if __name__ == "__main__":
    main()