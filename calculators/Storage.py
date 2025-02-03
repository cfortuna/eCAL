from typing import Dict, Union
from enum import Enum

class RAIDLevel(Enum):
    NO_RAID = "NO_RAID"  # # Single disk, no overhead
    RAID0 = "RAID0" # Pure striping, no overhead
    RAID1 = "RAID1" # Full mirroring
    RAID5 = "RAID5" # One disk for parity
    RAID6 = "RAID6" # Two disks for parity
    RAID10 = "RAID10" # Always 50% usable space

class Storage:
    # ssd consumption 1.2 Wh/tb converted to joules per bit
    SSD_ENERGY_PER_BIT = 5.4*1e-10
    # hdd consumption 0.65 Wh/tb converted to joules per bit
    HDD_ENERGY_PER_BIT = 2.92*1e-10

    def __init__(self, storage_type: str = 'SSD', raid_level: str = 'NO_RAID', num_disks: int = 1):
        """
        Initialize storage calculator with specified type, RAID level, and number of disks
        
        Args:
            storage_type (str): 'SSD' or 'HDD'
            raid_level (str): RAID level (NO_RAID, RAID0, RAID1, RAID5, RAID6, RAID10)
            num_disks (int): Number of disks in the array (1 for NO_RAID)
        """
        self.storage_type = storage_type.upper()
        self.raid_level = RAIDLevel(raid_level)
        
        # Validate minimum disk requirements
        min_disks = {
            RAIDLevel.NO_RAID: 1,  # Single disk
            RAIDLevel.RAID0: 2,
            RAIDLevel.RAID1: 2,
            RAIDLevel.RAID5: 3,
            RAIDLevel.RAID6: 4,
            RAIDLevel.RAID10: 4
        }
        
        if num_disks < min_disks[self.raid_level]:
            raise ValueError(f"{raid_level} requires minimum {min_disks[self.raid_level]} disks")
        
        # Force num_disks to 1 for NO_RAID
        if self.raid_level == RAIDLevel.NO_RAID:
            self.num_disks = 1
        else:
            self.num_disks = num_disks

    def get_raid_factor(self) -> float:
        """Calculate storage multiplication factor based on RAID level and number of disks"""
        
        if self.raid_level == RAIDLevel.NO_RAID:
            return 1.0  # Single disk, no overhead
            
        elif self.raid_level == RAIDLevel.RAID0:
            return 1.0  # Pure striping, no overhead
            
        elif self.raid_level == RAIDLevel.RAID1:
            return 2.0  # Full mirroring
            
        elif self.raid_level == RAIDLevel.RAID5:
            # One disk for parity
            return self.num_disks / (self.num_disks - 1)
            
        elif self.raid_level == RAIDLevel.RAID6:
            # Two disks for parity
            if self.num_disks <= 4:
                return 2.0  # With 4 disks, 50% overhead
            return self.num_disks / (self.num_disks - 2)
            
        elif self.raid_level == RAIDLevel.RAID10:
            return 2.0  # Always 50% usable space
            
        raise ValueError(f"Unknown RAID level: {self.raid_level}")

    def get_write_pattern(self) -> float:
        """Calculate write multiplication factor based on RAID level and number of disks"""
        
        if self.raid_level == RAIDLevel.NO_RAID:
            return 1.0  # Single write
            
        elif self.raid_level == RAIDLevel.RAID0:
            return 1.0  # Single write
            
        elif self.raid_level == RAIDLevel.RAID1:
            return float(self.num_disks)  # Write to all mirrors
            
        elif self.raid_level == RAIDLevel.RAID5:
            # Write data + calculate and write parity
            return 2.0
            
        elif self.raid_level == RAIDLevel.RAID6:
            # Write data + calculate and write two parities
            return 3.0
            
        elif self.raid_level == RAIDLevel.RAID10:
            return 2.0  # Write to primary and mirror
            
        raise ValueError(f"Unknown RAID level: {self.raid_level}")

    def calculate_energy(self, data_bits: int) -> Dict[str, Union[float, Dict]]:
        """
        Calculate energy consumption for storing data with specified RAID level
        
        Args:
            data_bits (int): Amount of data in bits
            
        Returns:
            Dict containing total energy and detailed breakdown
        """
        if self.storage_type == 'SSD':
            energy_per_bit = self.SSD_ENERGY_PER_BIT
        elif self.storage_type == 'HDD':
            energy_per_bit = self.HDD_ENERGY_PER_BIT
        else:
            raise ValueError(f"Invalid storage type: {self.storage_type}")

        raid_factor = self.get_raid_factor()
        write_pattern = self.get_write_pattern()
        
        # Calculate raw storage required (including RAID overhead)
        raw_storage = data_bits * raid_factor
        
        # # Calculate write energy (based on write patterns)
        write_energy = data_bits * energy_per_bit * write_pattern
        
        # Calculate ongoing storage energy (based on raw storage)
        storage_energy = raw_storage * energy_per_bit

        return {
            'total_energy': storage_energy,
            'details': {
                'raw_storage_bits': raw_storage,
                'write_energy': write_energy,
                'storage_energy': storage_energy,
                'raid_factor': raid_factor,
                'write_pattern': write_pattern,
                'usable_capacity_percentage': (1/raid_factor) * 100,
                'num_disks': self.num_disks
            }
        }

# Example usage:
if __name__ == "__main__":
    num_bits = 256*64  # Ns 256 double precision 
    
    print("\nComparing different storage configurations with minimum number of disks:")
    
    # Test single disk (NO_RAID)
    # storage = Storage('HDD', 'NO_RAID')
    # result = storage.calculate_energy(num_bits)
    # print(f"\nSingle SSD (NO_RAID):")
    # print(f"Storage Efficiency: {result['details']['usable_capacity_percentage']:.1f}%")
    # print(f"Total Energy: {result['total_energy']} J")
       # all raid levels with minimum number of disks
    min_disks = {
        RAIDLevel.NO_RAID: 1,
        RAIDLevel.RAID0: 2,
        RAIDLevel.RAID1: 2,
        RAIDLevel.RAID5: 3,
        RAIDLevel.RAID6: 4,
        RAIDLevel.RAID10: 4
    }
    # all raid levels with minimum number of disks
    print("#########################################HDD#########################################")
    for raid_level in RAIDLevel:
        storage = Storage('HDD', raid_level, min_disks[raid_level])
        result = storage.calculate_energy(num_bits)
        print(f"\n{raid_level}:")
        print(f"Storage Efficiency: {result['details']['usable_capacity_percentage']:.1f}%")
        print(f"Total Energy: {result['total_energy']} J")

    print("#########################################SSD#########################################")
    for raid_level in RAIDLevel:
        storage = Storage('SSD', raid_level, min_disks[raid_level])
        result = storage.calculate_energy(num_bits)
        print(f"\n{raid_level}:")
        print(f"Storage Efficiency: {result['details']['usable_capacity_percentage']:.1f}%")
        print(f"Total Energy: {result['total_energy']} J")

