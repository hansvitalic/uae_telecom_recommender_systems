"""
Configuration Management for UAE Telecom Recommender Systems

Handles configuration settings, sector parameters, and system customization.
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from ..models.sector import TelecomSectorType


@dataclass
class RecommenderConfig:
    """Main configuration for the recommender system"""
    
    # Algorithm weights
    collaborative_filtering_weight: float = 0.6
    content_based_weight: float = 0.4
    risk_assessment_weight: float = 0.8
    
    # Risk thresholds
    high_risk_threshold: float = 4.0
    medium_risk_threshold: float = 3.0
    
    # Recommendation limits
    max_recommendations_per_project: int = 15
    max_similar_projects: int = 5
    
    # UAE-specific settings
    default_regulatory_body: str = "TDRA"
    default_currency: str = "AED"
    major_emirates: list = None
    
    # ML model parameters
    svd_components: int = 50
    clustering_clusters: int = 5
    
    # Performance settings
    cache_enabled: bool = True
    cache_timeout_minutes: int = 30
    
    def __post_init__(self):
        if self.major_emirates is None:
            self.major_emirates = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"]


@dataclass
class SectorConfig:
    """Sector-specific configuration"""
    
    sector_type: TelecomSectorType
    risk_weights: Dict[str, float]
    performance_thresholds: Dict[str, float]
    regulatory_requirements: list
    best_practices: list
    typical_project_duration_months: int
    budget_range_aed: tuple


class ConfigManager:
    """Manages configuration for the recommender system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.sector_configs = self._load_sector_configs()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return str(Path(__file__).parent / "default_config.yaml")
    
    def _load_config(self) -> RecommenderConfig:
        """Load main configuration"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                return RecommenderConfig(**config_dict.get('recommender', {}))
            else:
                return RecommenderConfig()
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return RecommenderConfig()
    
    def _load_sector_configs(self) -> Dict[TelecomSectorType, SectorConfig]:
        """Load sector-specific configurations"""
        configs = {}
        
        # Mobile Networks configuration
        configs[TelecomSectorType.MOBILE_NETWORKS] = SectorConfig(
            sector_type=TelecomSectorType.MOBILE_NETWORKS,
            risk_weights={
                "technical": 0.25,
                "regulatory": 0.20,
                "financial": 0.15,
                "operational": 0.15,
                "security": 0.15,
                "market": 0.10
            },
            performance_thresholds={
                "network_coverage": 0.95,
                "call_success_rate": 0.98,
                "data_throughput_mbps": 100,
                "latency_ms": 20
            },
            regulatory_requirements=[
                "5G spectrum license",
                "Network security certification",
                "Emergency services integration",
                "Quality of service compliance"
            ],
            best_practices=[
                "Network slicing implementation",
                "AI-driven optimization",
                "Edge computing deployment",
                "Massive MIMO technology"
            ],
            typical_project_duration_months=18,
            budget_range_aed=(10_000_000, 500_000_000)
        )
        
        # Fiber Optic configuration
        configs[TelecomSectorType.FIBER_OPTIC] = SectorConfig(
            sector_type=TelecomSectorType.FIBER_OPTIC,
            risk_weights={
                "operational": 0.30,
                "regulatory": 0.25,
                "technical": 0.20,
                "financial": 0.15,
                "environmental": 0.10
            },
            performance_thresholds={
                "fiber_penetration": 0.80,
                "installation_speed_km_per_month": 50,
                "signal_loss_db_per_km": 0.3,
                "availability": 0.9999
            },
            regulatory_requirements=[
                "Municipal permits",
                "Right-of-way approvals",
                "Environmental compliance",
                "Safety certifications"
            ],
            best_practices=[
                "Micro-trenching techniques",
                "FTTH architecture",
                "Smart fiber management",
                "Preventive maintenance"
            ],
            typical_project_duration_months=24,
            budget_range_aed=(50_000_000, 1_000_000_000)
        )
        
        # IoT M2M configuration
        configs[TelecomSectorType.IOT_M2M] = SectorConfig(
            sector_type=TelecomSectorType.IOT_M2M,
            risk_weights={
                "security": 0.30,
                "technical": 0.25,
                "market": 0.20,
                "regulatory": 0.15,
                "operational": 0.10
            },
            performance_thresholds={
                "device_connectivity": 0.99,
                "message_delivery_rate": 0.98,
                "battery_life_years": 5,
                "platform_scalability_devices": 1_000_000
            },
            regulatory_requirements=[
                "Device type approval",
                "Spectrum allocation",
                "Data privacy compliance",
                "Cybersecurity standards"
            ],
            best_practices=[
                "LoRaWAN deployment",
                "Device lifecycle management",
                "Security by design",
                "Edge analytics"
            ],
            typical_project_duration_months=8,
            budget_range_aed=(2_000_000, 50_000_000)
        )
        
        return configs
    
    def get_config(self) -> RecommenderConfig:
        """Get main configuration"""
        return self.config
    
    def get_sector_config(self, sector_type: TelecomSectorType) -> Optional[SectorConfig]:
        """Get configuration for specific sector"""
        return self.sector_configs.get(sector_type)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save configuration to file"""
        save_path = path or self.config_path
        config_dict = {
            'recommender': asdict(self.config),
            'sectors': {
                sector.value: asdict(config) 
                for sector, config in self.sector_configs.items()
            }
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def get_uae_market_parameters(self) -> Dict[str, Any]:
        """Get UAE market-specific parameters"""
        return {
            "major_operators": ["Etisalat", "du", "Virgin Mobile UAE"],
            "regulatory_body": "TDRA",
            "currency": "AED",
            "emirates": self.config.major_emirates,
            "business_hours": "Sunday to Thursday, 8 AM to 6 PM",
            "official_languages": ["Arabic", "English"],
            "time_zone": "UTC+4",
            "peak_seasons": {
                "ramadan": "Variable (lunar calendar)",
                "summer": "June to September",
                "expo_season": "October to March"
            },
            "weather_considerations": {
                "sandstorms": "March to May",
                "extreme_heat": "June to September",
                "humidity": "Year-round consideration"
            }
        }
    
    def get_risk_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get risk assessment matrix"""
        return {
            "probability_levels": {
                "very_low": 0.1,
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "very_high": 0.9
            },
            "impact_multipliers": {
                "very_low": 1.0,
                "low": 2.0,
                "medium": 3.0,
                "high": 4.0,
                "very_high": 5.0
            },
            "sector_multipliers": {
                TelecomSectorType.MOBILE_NETWORKS.value: 1.2,
                TelecomSectorType.FIBER_OPTIC.value: 1.0,
                TelecomSectorType.IOT_M2M.value: 1.1
            }
        }
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {}
        
        # Check weight sums
        total_weight = (self.config.collaborative_filtering_weight + 
                       self.config.content_based_weight)
        validation_results["weights_sum_valid"] = abs(total_weight - 1.0) < 0.01
        
        # Check thresholds
        validation_results["risk_thresholds_valid"] = (
            0 < self.config.medium_risk_threshold < self.config.high_risk_threshold <= 5.0
        )
        
        # Check ML parameters
        validation_results["ml_params_valid"] = (
            self.config.svd_components > 0 and 
            self.config.clustering_clusters > 0
        )
        
        # Check UAE parameters
        validation_results["uae_params_valid"] = (
            len(self.config.major_emirates) == 7 and
            self.config.default_currency == "AED"
        )
        
        return validation_results


# Default configuration template
DEFAULT_CONFIG_TEMPLATE = """
recommender:
  collaborative_filtering_weight: 0.6
  content_based_weight: 0.4
  risk_assessment_weight: 0.8
  high_risk_threshold: 4.0
  medium_risk_threshold: 3.0
  max_recommendations_per_project: 15
  max_similar_projects: 5
  default_regulatory_body: "TDRA"
  default_currency: "AED"
  major_emirates:
    - "Dubai"
    - "Abu Dhabi"
    - "Sharjah"
    - "Ajman"
    - "Ras Al Khaimah"
    - "Fujairah"
    - "Umm Al Quwain"
  svd_components: 50
  clustering_clusters: 5
  cache_enabled: true
  cache_timeout_minutes: 30

sectors:
  mobile_networks:
    risk_weights:
      technical: 0.25
      regulatory: 0.20
      financial: 0.15
      operational: 0.15
      security: 0.15
      market: 0.10
    performance_thresholds:
      network_coverage: 0.95
      call_success_rate: 0.98
      data_throughput_mbps: 100
      latency_ms: 20
    typical_project_duration_months: 18
    budget_range_aed: [10000000, 500000000]
  
  fiber_optic:
    risk_weights:
      operational: 0.30
      regulatory: 0.25
      technical: 0.20
      financial: 0.15
      environmental: 0.10
    performance_thresholds:
      fiber_penetration: 0.80
      installation_speed_km_per_month: 50
      signal_loss_db_per_km: 0.3
      availability: 0.9999
    typical_project_duration_months: 24
    budget_range_aed: [50000000, 1000000000]
"""


def create_default_config_file(path: str) -> None:
    """Create a default configuration file"""
    with open(path, 'w') as f:
        f.write(DEFAULT_CONFIG_TEMPLATE)


def load_config_from_dict(config_dict: Dict[str, Any]) -> RecommenderConfig:
    """Load configuration from dictionary"""
    return RecommenderConfig(**config_dict.get('recommender', {}))


def export_config_to_json(config: RecommenderConfig, path: str) -> None:
    """Export configuration to JSON format"""
    with open(path, 'w') as f:
        json.dump(asdict(config), f, indent=2)