// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AddressBook {
    // State variables
    mapping(address => address[]) private addresses;
    mapping(address => mapping(address => string)) private aliases;
    mapping(address => mapping(address => bool)) private addressExists;
    
    // Events
    event AddressAdded(address indexed owner, address indexed addr, string alias);
    
    // Custom errors
    error AddressAlreadyExists();
    error EmptyAddress();
    error EmptyAlias();
    error InvalidAddress();
    
    // Constants
    uint256 private constant MAX_ADDRESSES = 100;
    uint256 private constant MAX_ALIAS_LENGTH = 32;

    /**
     * @notice Adds a new address with an alias
     * @param addr The address to add
     * @param alias The alias for the address
     */
    function addAddress(address addr, string calldata alias) public {
        // Input validation
        if (addr == address(0)) revert EmptyAddress();
        if (bytes(alias).length == 0) revert EmptyAlias();
        if (bytes(alias).length > MAX_ALIAS_LENGTH) revert InvalidAddress();
        if (addressExists[msg.sender][addr]) revert AddressAlreadyExists();
        if (addresses[msg.sender].length >= MAX_ADDRESSES) revert InvalidAddress();
        
        // Update state
        addresses[msg.sender].push(addr);
        aliases[msg.sender][addr] = alias;
        addressExists[msg.sender][addr] = true;
        
        // Emit event
        emit AddressAdded(msg.sender, addr, alias);
    }

    /**
     * @notice Retrieves all addresses for a given owner
     * @param owner The address of the owner
     * @return Array of addresses
     */
    function getAddressArray(address owner) public view returns (address[] memory) {
        return addresses[owner];
    }

    /**
     * @notice Gets the alias for a specific address
     * @param addrOwner The owner of the address book entry
     * @param addr The address to look up
     * @return The alias string
     */
    function getAlias(address addrOwner, address addr) public view returns (string memory) {
        return aliases[addrOwner][addr];
    }
}