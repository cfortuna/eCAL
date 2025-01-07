from typing import Dict, Union
from protocol_configs import (
    LayerProtocol, APPLICATION_PROTOCOLS, PRESENTATION_PROTOCOLS,
    SESSION_PROTOCOLS, TRANSPORT_PROTOCOLS, NETWORK_PROTOCOLS,
    DATALINK_PROTOCOLS, PHYSICAL_PROTOCOLS
)

class Transmission_simple:
    """
    Simplified calculator for network energy consumption that allows protocol selection
    for each OSI layer, focusing only on data and control plane overheads
    """
    
    def __init__(self, 
                 application: str = 'HTTP',
                 presentation: str = 'TLS',
                 session: str = 'RPC',
                 transport: str = 'TCP',
                 network: str = 'IPv4',
                 datalink: str = 'ETHERNET',
                 physical: str = 'WIFI_PHY'):
        """
        Initialize calculator with specific protocols for each layer
        """
        self.protocols = {
            'application': APPLICATION_PROTOCOLS[application],
            'presentation': PRESENTATION_PROTOCOLS[presentation],
            'session': SESSION_PROTOCOLS[session],
            'transport': TRANSPORT_PROTOCOLS[transport],
            'network': NETWORK_PROTOCOLS[network],
            'datalink': DATALINK_PROTOCOLS[datalink],
            'physical': PHYSICAL_PROTOCOLS[physical]
        }
    
    def calculate_layer_energy(self, protocol: LayerProtocol, input_bits: int) -> Dict[str, Union[float, int]]:
        """Calculate energy consumption for a single layer"""
        
        # Calculate overhead bits
        data_plane_bits = int(input_bits * protocol.data_plane_overhead)
        control_plane_bits = int(input_bits * protocol.control_plane_overhead)
        
        # Total bits at this layer
        total_bits = input_bits + data_plane_bits + control_plane_bits
        
        # Calculate energy
        total_energy = total_bits * protocol.base_energy_per_bit
        
        return {
            'total_bits': total_bits,
            'energy': total_energy,
            'breakdown': {
                'data_plane_overhead': data_plane_bits,
                'control_plane_overhead': control_plane_bits,
                'data_plane_energy': data_plane_bits * protocol.base_energy_per_bit,
                'control_plane_energy': control_plane_bits * protocol.base_energy_per_bit
            }
        }
    
    def calculate_energy(self, data_bits: int) -> Dict[str, Union[float, Dict]]:
        """
        Calculate energy consumption across all layers
        
        Args:
            data_bits: Original data bits to transmit
            
        Returns:
            Dictionary containing energy calculations and breakdown by layer
        """
        current_bits = data_bits
        total_energy = 0
        layer_results = {}
        
        # Process each layer from application to physical
        for layer_name, protocol in self.protocols.items():
            layer_result = self.calculate_layer_energy(protocol, current_bits)
            
            layer_results[layer_name] = {
                'protocol': protocol.name,
                'energy': layer_result['energy'],
                'breakdown': layer_result['breakdown']
            }
            
            total_energy += layer_result['energy']
            current_bits = layer_result['total_bits']
        
        return {
            'total_energy': total_energy,
            'total_bits': current_bits,
            'original_bits': data_bits,
            'layer_breakdown': layer_results
        }

# Usage example
if __name__ == "__main__":
    # Create calculator with custom protocol stack
    calculator = Transmission_simple(
        application='HTTP',
        presentation='TLS',
        session='RPC',
        transport='TCP',
        network='IPv4',
        datalink='ETHERNET',
        physical='WIFI_PHY'
    )
    
    # Calculate energy for 1MB of data
    data_size_bits = 1024 * 1024 * 8  # 1MB in bits
    result = calculator.calculate_energy(data_size_bits)
    
    print(f"Total Energy: {result['total_energy']} Joules")
    print("\nBreakdown by layer:")
    for layer, data in result['layer_breakdown'].items():
        print(f"{layer} ({data['protocol']}): {data['energy']} Joules") 