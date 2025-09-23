"""
Main RMS Pipeline Orchestrator
Coordinates the execution of all RMS modules and ML analysis
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import RMS modules
from rms_pipeline.risk_identification import RiskIdentifier
from rms_pipeline.qualitative_analysis import QualitativeAnalyzer
from rms_pipeline.quantitative_analysis import QuantitativeAnalyzer
from rms_pipeline.response_planning import ResponsePlanner
from rms_pipeline.monitoring import RiskMonitor
from rms_pipeline.documentation import RMSDocumentationGenerator

# Import ML modules
from ml_models.feature_engineering import FeatureEngineer
from ml_models.risk_predictor import RiskPredictor

import config

class RMSPipelineOrchestrator:
    """
    Main orchestrator for the RMS pipeline and ML analysis
    """
    
    def __init__(self):
        """Initialize the pipeline orchestrator"""
        self.execution_start_time = datetime.now()
        self.pipeline_results = {}
        self.ml_results = {}
        self.risks_df = None
        
        # Ensure output directory exists
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        
        print("🚀 Initializing UAE Telecom RMS Pipeline")
        print("=" * 60)
    
    def load_risk_data(self) -> pd.DataFrame:
        """Load the risk register data"""
        try:
            self.risks_df = pd.read_csv(config.RISK_REGISTER_CSV)
            print(f"✓ Loaded {len(self.risks_df)} risks from register")
            print(f"✓ Data columns: {self.risks_df.shape[1]} fields")
            return self.risks_df
        except Exception as e:
            print(f"❌ Error loading risk data: {e}")
            raise
    
    def execute_rms_pipeline(self) -> Dict[str, Any]:
        """Execute the complete RMS pipeline"""
        print("\n🔍 EXECUTING RMS PIPELINE")
        print("=" * 60)
        
        # Load risk data
        if self.risks_df is None:
            self.load_risk_data()
        
        pipeline_results = {}
        
        # Step 1: Risk Identification
        print("\n1️⃣ RISK IDENTIFICATION")
        print("-" * 40)
        identifier = RiskIdentifier()
        identification_report = identifier.generate_identification_report()
        pipeline_results['identification'] = identification_report
        
        # Step 2: Qualitative Analysis
        print("\n2️⃣ QUALITATIVE ANALYSIS")
        print("-" * 40)
        qualitative_analyzer = QualitativeAnalyzer(self.risks_df)
        qualitative_report = qualitative_analyzer.generate_qualitative_report()
        pipeline_results['qualitative_analysis'] = qualitative_report
        
        # Step 3: Quantitative Analysis
        print("\n3️⃣ QUANTITATIVE ANALYSIS")
        print("-" * 40)
        quantitative_analyzer = QuantitativeAnalyzer(self.risks_df)
        quantitative_report = quantitative_analyzer.generate_quantitative_report()
        pipeline_results['quantitative_analysis'] = quantitative_report
        
        # Step 4: Response Planning
        print("\n4️⃣ RESPONSE PLANNING")
        print("-" * 40)
        response_planner = ResponsePlanner(self.risks_df, pipeline_results)
        response_report = response_planner.generate_response_report()
        pipeline_results['response_planning'] = response_report
        
        # Step 5: Risk Monitoring
        print("\n5️⃣ RISK MONITORING")
        print("-" * 40)
        risk_monitor = RiskMonitor(self.risks_df)
        monitoring_report = risk_monitor.generate_monitoring_report()
        pipeline_results['monitoring'] = monitoring_report
        
        # Step 6: Documentation
        print("\n6️⃣ DOCUMENTATION")
        print("-" * 40)
        doc_generator = RMSDocumentationGenerator(self.risks_df, pipeline_results)
        documentation_report = doc_generator.generate_documentation_report()
        pipeline_results['documentation'] = documentation_report
        
        self.pipeline_results = pipeline_results
        
        print("\n✅ RMS PIPELINE COMPLETED SUCCESSFULLY")
        return pipeline_results
    
    def execute_ml_analysis(self) -> Dict[str, Any]:
        """Execute the ML predictive analytics"""
        print("\n🤖 EXECUTING ML PREDICTIVE ANALYTICS")
        print("=" * 60)
        
        ml_results = {}
        
        # Step 1: Feature Engineering
        print("\n1️⃣ FEATURE ENGINEERING")
        print("-" * 40)
        feature_engineer = FeatureEngineer(self.risks_df)
        ml_dataset, feature_metadata = feature_engineer.prepare_ml_dataset()
        dataset_path = feature_engineer.export_ml_dataset()
        
        ml_results['feature_engineering'] = {
            'dataset_path': dataset_path,
            'metadata': feature_metadata,
            'completion_status': 'Feature Engineering Complete'
        }
        
        # Step 2: ML Model Training and Prediction
        print("\n2️⃣ ML MODEL TRAINING & PREDICTION")
        print("-" * 40)
        risk_predictor = RiskPredictor(dataset_path)
        ml_report = risk_predictor.generate_ml_report()
        ml_results['model_analysis'] = ml_report
        
        self.ml_results = ml_results
        
        print("\n✅ ML ANALYSIS COMPLETED SUCCESSFULLY")
        return ml_results
    
    def export_rms_processed_data(self) -> str:
        """Export processed RMS data"""
        if self.risks_df is None or not self.pipeline_results:
            print("❌ No RMS pipeline results to export")
            return None
        
        # Create enhanced risk register with analysis results
        enhanced_df = self.risks_df.copy()
        
        # Add qualitative analysis results
        if 'qualitative_analysis' in self.pipeline_results:
            qual_analysis = self.pipeline_results['qualitative_analysis']
            if 'prioritized_risks' in qual_analysis:
                # Create priority mapping
                priority_map = {
                    risk['Risk ID']: {
                        'Enhanced_Score': risk['Enhanced Score'],
                        'Priority_Level': risk['Priority Level']
                    }
                    for risk in qual_analysis['prioritized_risks']
                }
                
                # Add priority columns
                enhanced_df['Enhanced_Risk_Score'] = enhanced_df['Risk ID'].map(
                    lambda x: priority_map.get(x, {}).get('Enhanced_Score', 0)
                )
                enhanced_df['Priority_Level'] = enhanced_df['Risk ID'].map(
                    lambda x: priority_map.get(x, {}).get('Priority_Level', 'Unknown')
                )
        
        # Add quantitative analysis results
        if 'quantitative_analysis' in self.pipeline_results:
            quant_analysis = self.pipeline_results['quantitative_analysis']
            if 'risk_exposure' in quant_analysis and 'exposure_data' in quant_analysis['risk_exposure']:
                exposure_df = quant_analysis['risk_exposure']['exposure_data']
                
                # Merge exposure data
                enhanced_df = enhanced_df.merge(
                    exposure_df[['Risk ID', 'Probability %', 'Impact Cost', 'Expected Monetary Value']],
                    on='Risk ID',
                    how='left'
                )
        
        # Add response planning results
        if 'response_planning' in self.pipeline_results:
            response_analysis = self.pipeline_results['response_planning']
            if 'response_strategies' in response_analysis and 'response_strategies' in response_analysis['response_strategies']:
                strategies_list = response_analysis['response_strategies']['response_strategies']
                
                # Create strategy mapping
                strategy_map = {
                    strategy['Risk ID']: {
                        'Recommended_Strategy_Type': strategy['Recommended Strategy Type'],
                        'Enhanced_Strategy': strategy['Enhanced Strategy'],
                        'Response_Priority': strategy['Priority Level']
                    }
                    for strategy in strategies_list
                }
                
                # Add strategy columns
                enhanced_df['Recommended_Strategy_Type'] = enhanced_df['Risk ID'].map(
                    lambda x: strategy_map.get(x, {}).get('Recommended_Strategy_Type', 'Unknown')
                )
                enhanced_df['Enhanced_Mitigation_Strategy'] = enhanced_df['Risk ID'].map(
                    lambda x: strategy_map.get(x, {}).get('Enhanced_Strategy', '')
                )
                enhanced_df['Response_Priority_Level'] = enhanced_df['Risk ID'].map(
                    lambda x: strategy_map.get(x, {}).get('Response_Priority', 'Unknown')
                )
        
        # Add RMS processing timestamp
        enhanced_df['RMS_Processing_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        enhanced_df['RMS_Pipeline_Version'] = '1.0'
        
        # Export enhanced risk register
        output_path = config.RMS_OUTPUT_FILE
        enhanced_df.to_csv(output_path, index=False)
        
        print(f"✓ RMS processed data exported to: {output_path}")
        return str(output_path)
    
    def generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution summary"""
        execution_end_time = datetime.now()
        execution_duration = execution_end_time - self.execution_start_time
        
        summary = {
            "pipeline_metadata": {
                "execution_start": self.execution_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "execution_end": execution_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                "execution_duration": str(execution_duration),
                "total_risks_processed": len(self.risks_df) if self.risks_df is not None else 0,
                "pipeline_version": "1.0"
            },
            "rms_pipeline_summary": self._summarize_rms_results(),
            "ml_analysis_summary": self._summarize_ml_results(),
            "deliverables": self._list_deliverables(),
            "key_insights": self._extract_key_insights(),
            "recommendations": self._compile_recommendations(),
            "next_steps": self._define_next_steps()
        }
        
        return summary
    
    def _summarize_rms_results(self) -> Dict[str, Any]:
        """Summarize RMS pipeline results"""
        if not self.pipeline_results:
            return {"status": "Not executed"}
        
        summary = {
            "steps_completed": len(self.pipeline_results),
            "identification": {
                "risks_identified": self.pipeline_results.get('identification', {}).get('identification_summary', {}).get('total_risks_identified', 0),
                "data_quality_score": self.pipeline_results.get('identification', {}).get('identification_summary', {}).get('data_quality_score', 0)
            },
            "qualitative_analysis": {
                "critical_risks": self.pipeline_results.get('qualitative_analysis', {}).get('analysis_summary', {}).get('critical_risks', 0),
                "high_priority_risks": self.pipeline_results.get('qualitative_analysis', {}).get('analysis_summary', {}).get('high_priority_risks', 0)
            },
            "quantitative_analysis": {
                "total_risk_exposure": self.pipeline_results.get('quantitative_analysis', {}).get('analysis_summary', {}).get('total_risk_exposure', 0),
                "value_at_risk_95": self.pipeline_results.get('quantitative_analysis', {}).get('analysis_summary', {}).get('value_at_risk_95', 0)
            },
            "response_planning": {
                "strategies_developed": self.pipeline_results.get('response_planning', {}).get('planning_summary', {}).get('total_risks_planned', 0),
                "immediate_responses_needed": self.pipeline_results.get('response_planning', {}).get('planning_summary', {}).get('immediate_responses_needed', 0)
            },
            "monitoring": {
                "active_risks": self.pipeline_results.get('monitoring', {}).get('monitoring_summary', {}).get('active_risks', 0),
                "active_alerts": self.pipeline_results.get('monitoring', {}).get('monitoring_summary', {}).get('active_alerts', 0)
            }
        }
        
        return summary
    
    def _summarize_ml_results(self) -> Dict[str, Any]:
        """Summarize ML analysis results"""
        if not self.ml_results:
            return {"status": "Not executed"}
        
        summary = {
            "feature_engineering": {
                "total_features": self.ml_results.get('feature_engineering', {}).get('metadata', {}).get('total_features', 0),
                "dataset_shape": self.ml_results.get('feature_engineering', {}).get('metadata', {}).get('data_shape', (0, 0))
            },
            "model_analysis": {
                "models_trained": self.ml_results.get('model_analysis', {}).get('ml_summary', {}).get('total_models_trained', 0),
                "prediction_tasks": len(self.ml_results.get('model_analysis', {}).get('ml_summary', {}).get('prediction_tasks', []))
            }
        }
        
        return summary
    
    def _list_deliverables(self) -> Dict[str, str]:
        """List all pipeline deliverables"""
        deliverables = {
            "rms_processed_data": str(config.RMS_OUTPUT_FILE),
            "ml_ready_dataset": str(config.ML_DATASET_FILE),
            "comprehensive_documentation": str(config.OUTPUT_DIR / "comprehensive_rms_documentation.json")
        }
        
        # Add model files if they exist
        model_dir = config.OUTPUT_DIR / "models"
        if model_dir.exists():
            deliverables["ml_models_directory"] = str(model_dir)
        
        return deliverables
    
    def _extract_key_insights(self) -> List[str]:
        """Extract key insights from all analyses"""
        insights = []
        
        # RMS insights
        if self.pipeline_results:
            # Critical risks insight
            critical_count = self.pipeline_results.get('qualitative_analysis', {}).get('analysis_summary', {}).get('critical_risks', 0)
            if critical_count > 0:
                insights.append(f"Identified {critical_count} critical risks requiring immediate attention")
            
            # Risk exposure insight
            total_exposure = self.pipeline_results.get('quantitative_analysis', {}).get('analysis_summary', {}).get('total_risk_exposure', 0)
            if total_exposure > 0:
                insights.append(f"Total risk exposure estimated at ${total_exposure:,.2f}")
            
            # Response planning insight
            immediate_responses = self.pipeline_results.get('response_planning', {}).get('planning_summary', {}).get('immediate_responses_needed', 0)
            if immediate_responses > 0:
                insights.append(f"{immediate_responses} risks require immediate response implementation")
        
        # ML insights
        if self.ml_results:
            feature_count = self.ml_results.get('feature_engineering', {}).get('metadata', {}).get('total_features', 0)
            if feature_count > 0:
                insights.append(f"Generated {feature_count} features for predictive modeling")
            
            model_count = self.ml_results.get('model_analysis', {}).get('ml_summary', {}).get('total_models_trained', 0)
            if model_count > 0:
                insights.append(f"Trained {model_count} ML models for risk prediction")
        
        return insights
    
    def _compile_recommendations(self) -> List[str]:
        """Compile recommendations from all modules"""
        recommendations = []
        
        # Collect recommendations from each module
        for module_name, module_results in self.pipeline_results.items():
            if 'recommendations' in module_results:
                module_recs = module_results['recommendations']
                if isinstance(module_recs, list):
                    recommendations.extend(module_recs)
        
        # Add ML recommendations
        if self.ml_results and 'model_analysis' in self.ml_results:
            ml_recs = self.ml_results['model_analysis'].get('recommendations', [])
            if isinstance(ml_recs, list):
                recommendations.extend(ml_recs)
        
        # Add strategic recommendations
        strategic_recs = [
            "Implement continuous risk monitoring using the established framework",
            "Use ML predictions to enhance early risk detection capabilities",
            "Regular update of risk assessment models with new project data",
            "Establish risk management KPIs based on pipeline insights"
        ]
        
        recommendations.extend(strategic_recs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _define_next_steps(self) -> List[str]:
        """Define immediate next steps"""
        next_steps = [
            "Review critical risks identified by the RMS pipeline",
            "Implement immediate response strategies for high-priority risks",
            "Deploy ML models for ongoing risk prediction and monitoring",
            "Establish regular risk review meetings using pipeline insights",
            "Integrate pipeline outputs with project management systems",
            "Schedule quarterly comprehensive risk assessments",
            "Train project teams on new risk management processes"
        ]
        
        return next_steps
    
    def export_final_outputs(self) -> Dict[str, str]:
        """Export all final outputs and reports"""
        print("\n📁 EXPORTING FINAL OUTPUTS")
        print("-" * 40)
        
        outputs = {}
        
        # Export RMS processed data
        rms_output_path = self.export_rms_processed_data()
        if rms_output_path:
            outputs['rms_processed_data'] = rms_output_path
        
        # Export pipeline summary
        summary = self.generate_pipeline_summary()
        summary_path = config.OUTPUT_DIR / "pipeline_execution_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        outputs['execution_summary'] = str(summary_path)
        
        print(f"✓ Pipeline summary exported to: {summary_path}")
        
        # List other outputs that should exist
        if config.ML_DATASET_FILE.exists():
            outputs['ml_dataset'] = str(config.ML_DATASET_FILE)
        
        documentation_file = config.OUTPUT_DIR / "comprehensive_rms_documentation.json"
        if documentation_file.exists():
            outputs['comprehensive_documentation'] = str(documentation_file)
        
        return outputs
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete RMS pipeline and ML analysis"""
        print("🎯 STARTING COMPLETE RMS PIPELINE EXECUTION")
        print("=" * 60)
        
        try:
            # Load data
            self.load_risk_data()
            
            # Execute RMS pipeline
            rms_results = self.execute_rms_pipeline()
            
            # Execute ML analysis
            ml_results = self.execute_ml_analysis()
            
            # Export final outputs
            final_outputs = self.export_final_outputs()
            
            # Generate final summary
            final_summary = self.generate_pipeline_summary()
            
            print("\n🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"✅ Total risks processed: {final_summary['pipeline_metadata']['total_risks_processed']}")
            print(f"✅ Execution duration: {final_summary['pipeline_metadata']['execution_duration']}")
            print(f"✅ RMS steps completed: {final_summary['rms_pipeline_summary']['steps_completed']}")
            print(f"✅ ML models trained: {final_summary['ml_analysis_summary']['model_analysis']['models_trained']}")
            print("\n📋 Key Deliverables:")
            for name, path in final_outputs.items():
                print(f"  • {name}: {path}")
            
            return {
                'execution_summary': final_summary,
                'rms_results': rms_results,
                'ml_results': ml_results,
                'final_outputs': final_outputs,
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            print(f"\n❌ PIPELINE EXECUTION FAILED")
            print(f"Error: {str(e)}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'execution_summary': self.generate_pipeline_summary()
            }

def main():
    """Main execution function"""
    # Create and run the pipeline orchestrator
    orchestrator = RMSPipelineOrchestrator()
    results = orchestrator.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    main()