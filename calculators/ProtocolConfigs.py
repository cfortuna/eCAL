from dataclasses import dataclass
# TODO add realistic values
@dataclass
class LayerProtocol:
    """Protocol metrics for a single layer"""
    name: str                    # Protocol name
    data_plane_overhead: float   # Data plane overhead ratio
    control_plane_overhead: float # Control plane overhead ratio
    base_energy_per_bit: float   # Energy consumption per bit

# Protocol configurations for each layer
APPLICATION_PROTOCOLS = {
    'HTTP': LayerProtocol(
        name='HTTP',
        data_plane_overhead=0.1,      # 5% headers and data formatting
        control_plane_overhead=0.05,    # 2% control messages
        base_energy_per_bit=0.00000001 # 10 nJ/bit
    ),
    'FTP': LayerProtocol(
        name='FTP',
        data_plane_overhead=0.03,
        control_plane_overhead=0.04,
        base_energy_per_bit=0.00000001
    ),
    'Generic_application': LayerProtocol(
        name='Generic_application',
        data_plane_overhead=0.1,
        control_plane_overhead=0.05,
        base_energy_per_bit=2.857142857142857e-09
    )
}

PRESENTATION_PROTOCOLS = {
    'TLS': LayerProtocol(
        name='TLS',
        data_plane_overhead=0.08,      # 8% encryption overhead
        control_plane_overhead=0.03,    # 3% handshake
        base_energy_per_bit=0.00000002 # 20 nJ/bit
    ),
    'SSL': LayerProtocol(
        name='SSL',
        data_plane_overhead=0.07,
        control_plane_overhead=0.04,
        base_energy_per_bit=0.00000002
    ),
    'Generic_presentation': LayerProtocol(
        name='Generic_presentation',
        data_plane_overhead=0.1,
        control_plane_overhead=0.05,
        base_energy_per_bit=2.857142857142857e-09
    )
}

SESSION_PROTOCOLS = {
    'RPC': LayerProtocol(
        name='RPC',
        data_plane_overhead=0.02,
        control_plane_overhead=0.02,
        base_energy_per_bit=0.00000001
    ),
    'Generic_session': LayerProtocol(
        name='Generic_session',
        data_plane_overhead=0.1,
        control_plane_overhead=0.05,
        base_energy_per_bit=2.857142857142857e-09
    )
}

TRANSPORT_PROTOCOLS = {
    'TCP': LayerProtocol(
        name='TCP',
        data_plane_overhead=0.05,      # 5% segmentation
        control_plane_overhead=0.10,    # 10% ACKs and control
        base_energy_per_bit=0.00000002 # 20 nJ/bit
    ),
    'UDP': LayerProtocol(
        name='UDP',
        data_plane_overhead=0.02,
        control_plane_overhead=0.01,
        base_energy_per_bit=0.00000001
    ),
    'Generic_transport': LayerProtocol(
        name='Generic_transport',
        data_plane_overhead=0.1,
        control_plane_overhead=0.05,
        base_energy_per_bit=2.857142857142857e-09
    )

}

NETWORK_PROTOCOLS = {
    'IPv4': LayerProtocol(
        name='IPv4',
        data_plane_overhead=0.03,
        control_plane_overhead=0.05,
        base_energy_per_bit=0.00000002
    ),
    'IPv6': LayerProtocol(
        name='IPv6',
        data_plane_overhead=0.04,
        control_plane_overhead=0.05,
        base_energy_per_bit=0.00000002
    ),
    'Generic_network': LayerProtocol(
        name='Generic_network',
        data_plane_overhead=0.1,
        control_plane_overhead=0.05,
        base_energy_per_bit=2.857142857142857e-09
    )
}

DATALINK_PROTOCOLS = {
    'ETHERNET': LayerProtocol(
        name='ETHERNET',
        data_plane_overhead=0.05,
        control_plane_overhead=0.05,
        base_energy_per_bit=0.00000003
    ),
    'WIFI_MAC': LayerProtocol(
        name='WIFI_MAC',
        data_plane_overhead=0.06,
        control_plane_overhead=0.08,
        base_energy_per_bit=0.00000004
    ),
    'Generic_datalink': LayerProtocol(
        name='Generic_datalink',
        data_plane_overhead=0.1,
        control_plane_overhead=0.05,
        base_energy_per_bit=2.857142857142857e-09
    )
}

PHYSICAL_PROTOCOLS = {
    'WIFI_PHY': LayerProtocol(
        name='WIFI_PHY',
        data_plane_overhead=0.10,
        control_plane_overhead=0.15,
        base_energy_per_bit=0.0000001
    ),
    'BLUETOOTH': LayerProtocol(
        name='BLUETOOTH',
        data_plane_overhead=0.08,
        control_plane_overhead=0.12,
        base_energy_per_bit=0.00000005
    ),
    'Generic_physical': LayerProtocol(
        name='Generic_physical',
        data_plane_overhead=0.1,
        control_plane_overhead=0.05,
        base_energy_per_bit=2.857142857142857e-09
    )
} 