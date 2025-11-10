// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract HotelRoom is ReentrancyGuard, Ownable {
    struct Room {
        uint256 id;
        uint256 agreementNumber;
        address tenant;
        bool isVacant;
        uint256 securityDeposit;
        uint256 rentPerMonth;
        uint256 lastRentPaid;
        uint256 rentDueDate;
    }

    mapping(uint256 => Room) public rooms;
    uint256 public roomCount;
    
    // Events
    event RoomAdded(uint256 indexed roomId, uint256 rent, uint256 deposit);
    event RoomRented(uint256 indexed roomId, address indexed tenant, uint256 agreementNumber);
    event RentPaid(uint256 indexed roomId, address indexed tenant, uint256 amount);
    event RoomVacated(uint256 indexed roomId, address indexed tenant);
    
    // Constants
    uint256 private constant SECONDS_PER_MONTH = 30 days;
    uint256 private constant MAXIMUM_DEPOSIT = 100 ether;
    uint256 private constant MAXIMUM_RENT = 50 ether;

    constructor() Ownable(msg.sender) {}

    function addRoom(uint256 rent, uint256 deposit) external onlyOwner {
        require(rent > 0 && rent <= MAXIMUM_RENT, "Invalid rent amount");
        require(deposit > 0 && deposit <= MAXIMUM_DEPOSIT, "Invalid deposit amount");
        
        roomCount++;
        rooms[roomCount] = Room({
            id: roomCount,
            agreementNumber: 0,
            tenant: address(0),
            isVacant: true,
            securityDeposit: deposit,
            rentPerMonth: rent,
            lastRentPaid: 0,
            rentDueDate: 0
        });
        
        emit RoomAdded(roomCount, rent, deposit);
    }

    function rentRoom(uint256 roomId) external payable nonReentrant {
        require(roomId > 0 && roomId <= roomCount, "Invalid room ID");
        Room storage room = rooms[roomId];
        require(room.isVacant, "Room is occupied");
        require(msg.value == room.securityDeposit + room.rentPerMonth, "Incorrect payment amount");
        
        room.tenant = msg.sender;
        room.isVacant = false;
        room.agreementNumber++;
        room.lastRentPaid = block.timestamp;
        room.rentDueDate = block.timestamp + SECONDS_PER_MONTH;
        
        emit RoomRented(roomId, msg.sender, room.agreementNumber);
    }

    function payRent(uint256 roomId) external payable nonReentrant {
        Room storage room = rooms[roomId];
        require(msg.sender == room.tenant, "Only tenant can pay rent");
        require(!room.isVacant, "Room is vacant");
        require(msg.value == room.rentPerMonth, "Incorrect rent amount");
        require(block.timestamp >= room.lastRentPaid + (SECONDS_PER_MONTH / 2), "Rent paid too early");
        
        room.lastRentPaid = block.timestamp;
        room.rentDueDate = block.timestamp + SECONDS_PER_MONTH;
        
        emit RentPaid(roomId, msg.sender, msg.value);
    }

    function vacateRoom(uint256 roomId) external nonReentrant {
        Room storage room = rooms[roomId];
        require(msg.sender == room.tenant, "Only tenant can vacate");
        require(!room.isVacant, "Room already vacant");
        
        // Return security deposit if rent is up to date
        if (block.timestamp <= room.rentDueDate) {
            (bool success, ) = payable(room.tenant).call{value: room.securityDeposit}("");
            require(success, "Security deposit return failed");
        }
        
        room.isVacant = true;
        room.tenant = address(0);
        room.lastRentPaid = 0;
        room.rentDueDate = 0;
        
        emit RoomVacated(roomId, msg.sender);
    }

    function withdrawFunds() external onlyOwner nonReentrant {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        (bool success, ) = owner().call{value: balance}("");
        require(success, "Withdrawal failed");
    }

    // Prevent accidental ETH transfers
    receive() external payable {
        revert("Use rentRoom or payRent functions to send ETH");
    }
}